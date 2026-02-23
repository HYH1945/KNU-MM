#!/usr/bin/env python3
"""
음성 특성 분석 모듈
음성의 피치, 에너지, 속도 등을 분석하여 응급 상황 신뢰도 판정

사용법:
    analyzer = VoiceCharacteristicsAnalyzer()
    audio_features = analyzer.extract_features(audio_file)
    confidence = analyzer.calculate_confidence(audio_features, llm_analysis)
"""

import numpy as np
from typing import Dict, Any

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️  librosa가 설치되지 않았습니다. 음성 특성 분석이 제한됩니다.")

# 설정 관리자 임포트
try:
    from core.config_manager import get_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    def get_config(*keys, default=None):
        return default


class VoiceCharacteristicsAnalyzer:
    """음성 특성 분석기 - 응급 상황 신뢰도 판정"""
    
    def __init__(self):
        """음성 분석기 초기화"""
        if not LIBROSA_AVAILABLE:
            print("⚠️  음성 특성 분석을 위해 librosa 설치가 필요합니다:")
            print("   pip install librosa")
        
        # config에서 임계값 로드
        self._load_thresholds()
    
    def _load_thresholds(self):
        """config에서 임계값 로드"""
        va = 'voice_analysis'
        
        # 샘플링 레이트
        self.sample_rate = get_config(va, 'sample_rate', default=16000)
        
        # 피치 임계값
        self.pitch_high_threshold = get_config(va, 'pitch', 'high_threshold', default=250)
        self.pitch_variability_threshold = get_config(va, 'pitch', 'variability_threshold', default=50)
        
        # 에너지 임계값
        self.energy_normalization = get_config(va, 'energy', 'normalization_factor', default=0.5)
        self.energy_volatility_threshold = get_config(va, 'energy', 'volatility_threshold', default=0.3)
        
        # 음성 속도 임계값
        self.speech_rate_fast_threshold = get_config(va, 'speech_rate', 'fast_threshold', default=6)
        
        # 유성음 비율 임계값
        self.voiced_ratio_low_threshold = get_config(va, 'voiced_ratio', 'low_threshold', default=0.3)
        
        # 지터/시머 임계값
        self.jitter_threshold = get_config(va, 'jitter_shimmer', 'jitter_threshold', default=0.1)
        self.shimmer_threshold = get_config(va, 'jitter_shimmer', 'shimmer_threshold', default=0.1)
        
        # 점수 계산 가중치
        self.llm_weight = get_config(va, 'scoring', 'llm_weight', default=0.6)
        self.voice_weight = get_config(va, 'scoring', 'voice_weight', default=0.4)
        
        # 우선순위 임계값
        self.priority_critical = get_config(va, 'priority_thresholds', 'critical', default=0.85)
        self.priority_high = get_config(va, 'priority_thresholds', 'high', default=0.65)
        self.priority_medium = get_config(va, 'priority_thresholds', 'medium', default=0.40)
    
    def extract_features(self, audio_file_path: str, sr: int = None) -> Dict[str, Any]:
        """
        오디오 파일에서 특성 추출
        
        Args:
            audio_file_path: 오디오 파일 경로
            sr: 샘플링 레이트 (None이면 config에서 로드)
        
        Returns:
            음성 특성 딕셔너리
        """
        if sr is None:
            sr = self.sample_rate
            
        if not LIBROSA_AVAILABLE:
            print("⚠️  librosa가 설치되지 않았습니다")
            return self._get_default_features()
        
        try:
            # 오디오 로드
            y, sr = librosa.load(audio_file_path, sr=sr)
            
            features = {
                'pitch': self._extract_pitch(y, sr),
                'energy': self._extract_energy(y),
                'speech_rate': self._estimate_speech_rate(y, sr),
                'spectral_characteristics': self._extract_spectral_features(y, sr),
                'voiced_unvoiced_ratio': self._analyze_voiced_unvoiced(y, sr),
                'jitter_shimmer': self._extract_jitter_shimmer(y, sr)
            }
            
            return features
        
        except Exception as e:
            print(f"⚠️  음성 특성 추출 실패: {e}")
            return self._get_default_features()
    
    def _extract_pitch(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """기본 주파수(Pitch) 추출"""
        try:
            # PYIN (음성 개선 YIN) 알고리즘으로 피치 추출
            f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            
            # NaN 제거
            f0_valid = f0[~np.isnan(f0)]
            
            if len(f0_valid) == 0:
                return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
            
            return {
                'mean': float(np.mean(f0_valid)),
                'std': float(np.std(f0_valid)),
                'min': float(np.min(f0_valid)),
                'max': float(np.max(f0_valid))
            }
        except Exception as e:
            print(f"⚠️  피치 추출 오류: {e}")
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
    
    def _extract_energy(self, y: np.ndarray) -> Dict[str, float]:
        """에너지(음량) 분석"""
        # RMS 에너지 계산
        frame_length = 2048
        hop_length = 512
        
        energy = np.sqrt(np.convolve(y**2, np.ones(frame_length)/frame_length, mode='valid')[::hop_length])
        
        return {
            'mean': float(np.mean(energy)),
            'std': float(np.std(energy)),
            'max': float(np.max(energy)),
            'min': float(np.min(energy))
        }
    
    def _estimate_speech_rate(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """음성 속도 추정 (음절/초)"""
        try:
            # 스펙트럼 중심(Spectral Centroid)과 onset 감지로 음절 추정
            try:
                # 새 버전의 librosa
                onsets = librosa.onset.onset_detect(y=y, sr=sr)
            except AttributeError:
                # 구 버전의 librosa
                onsets = librosa.onset.detect(y, sr=sr)
            
            duration = len(y) / sr
            
            # 대략적인 음절 수 (onset 기준)
            estimated_syllables = len(onsets)
            speech_rate = estimated_syllables / duration if duration > 0 else 0
            
            return {
                'estimated_syllables_per_second': float(speech_rate),
                'total_estimated_syllables': int(estimated_syllables),
                'duration_seconds': float(duration)
            }
        except Exception as e:
            print(f"⚠️  음성 속도 추정 오류: {e}")
            return {
                'estimated_syllables_per_second': 0,
                'total_estimated_syllables': 0,
                'duration_seconds': 0
            }
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """스펙트럼 특성 추출"""
        # 스펙트럼 중심
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        
        # 스펙트럼 롤오프
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # MFCC (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),
            'spectral_centroid_std': float(np.std(spectral_centroid)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_rolloff_std': float(np.std(spectral_rolloff)),
            'mfcc_mean': float(np.mean(mfcc))
        }
    
    def _analyze_voiced_unvoiced(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """유성음/무성음 비율 분석"""
        try:
            # 에너지 기반 유성음 감지
            frame_length = 2048
            hop_length = 512
            
            # 프레임별 에너지
            try:
                # 새 버전 librosa: 2D 배열 반환
                energy_result = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
                if energy_result.ndim > 1:
                    energy_frames = energy_result[0]
                else:
                    energy_frames = energy_result
            except AttributeError:
                # 구 버전 librosa: rmse 사용
                energy_frames = librosa.feature.rmse(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # 에너지 임계값 (평균의 0.3배)
            threshold = np.mean(energy_frames) * 0.3
            
            # 유성음 프레임 비율
            voiced_ratio = np.sum(energy_frames > threshold) / len(energy_frames)
            
            return {
                'voiced_ratio': float(voiced_ratio),
                'unvoiced_ratio': float(1.0 - voiced_ratio)
            }
        except Exception as e:
            print(f"⚠️  유성/무성음 분석 오류: {e}")
            return {'voiced_ratio': 0.5, 'unvoiced_ratio': 0.5}
    
    def _extract_jitter_shimmer(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """지터(Jitter), 시머(Shimmer) 추출 - 음성 품질"""
        try:
            # 간단한 지터/시머 추정
            # 실제 구현은 피치 추출 후 인접 피리오드 간 차이 계산
            
            f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
            f0_valid = f0[~np.isnan(f0)]
            
            if len(f0_valid) < 2:
                return {'jitter': 0.0, 'shimmer': 0.0}
            
            # 지터: 주파수 변동성
            pitch_diffs = np.abs(np.diff(f0_valid))
            jitter = np.mean(pitch_diffs) / np.mean(f0_valid) if np.mean(f0_valid) > 0 else 0
            
            # 시머: 에너지 변동성 (간단한 추정)
            energy = np.abs(y)
            energy_diffs = np.abs(np.diff(energy))
            shimmer = np.mean(energy_diffs) / np.mean(energy) if np.mean(energy) > 0 else 0
            
            return {
                'jitter': float(jitter),
                'shimmer': float(shimmer)
            }
        except Exception as e:
            print(f"⚠️  지터/시머 추출 오류: {e}")
            return {'jitter': 0.0, 'shimmer': 0.0}
    
    def _get_default_features(self) -> Dict[str, Any]:
        """기본 특성값 반환"""
        return {
            'pitch': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
            'energy': {'mean': 0, 'std': 0, 'max': 0, 'min': 0},
            'speech_rate': {'estimated_syllables_per_second': 0, 'total_estimated_syllables': 0, 'duration_seconds': 0},
            'spectral_characteristics': {'spectral_centroid_mean': 0, 'spectral_centroid_std': 0, 'spectral_rolloff_mean': 0, 'spectral_rolloff_std': 0, 'mfcc_mean': 0},
            'voiced_unvoiced_ratio': {'voiced_ratio': 0.5, 'unvoiced_ratio': 0.5},
            'jitter_shimmer': {'jitter': 0.0, 'shimmer': 0.0}
        }
    
    def analyze_emergency_indicators(self, audio_features: Dict[str, Any]) -> Dict[str, float]:
        """
        음성 특성으로부터 응급 신호 점수 계산
        
        Args:
            audio_features: 추출된 음성 특성
        
        Returns:
            각 지표별 신뢰도 점수 (0-1)
        """
        scores = {}
        
        # 1. 피치 기반 신호 (높은 피치 = 공포/긴장)
        pitch_mean = audio_features.get('pitch', {}).get('mean', 0)
        pitch_std = audio_features.get('pitch', {}).get('std', 0)
        
        # config에서 임계값 사용
        pitch_threshold = self.pitch_high_threshold
        variability_threshold = self.pitch_variability_threshold
        
        if pitch_mean > pitch_threshold:
            scores['high_pitch'] = min(1.0, (pitch_mean - pitch_threshold) / 100)
        else:
            scores['high_pitch'] = 0.0
        
        if pitch_std > variability_threshold:
            scores['pitch_variability'] = min(1.0, pitch_std / 100)
        else:
            scores['pitch_variability'] = pitch_std / variability_threshold * 0.3
        
        # 2. 에너지 기반 신호 (큰 음량 = 비명)
        energy_mean = audio_features.get('energy', {}).get('mean', 0)
        energy_std = audio_features.get('energy', {}).get('std', 0)
        
        # config에서 임계값 사용
        energy_norm = self.energy_normalization
        energy_vol = self.energy_volatility_threshold
        
        scores['high_energy'] = min(1.0, energy_mean / energy_norm)
        scores['energy_volatility'] = min(1.0, energy_std / energy_vol)
        
        # 3. 음성 속도 기반 신호 (빠른 속도 = 긴박)
        speech_rate = audio_features.get('speech_rate', {}).get('estimated_syllables_per_second', 0)
        
        # config에서 임계값 사용
        fast_threshold = self.speech_rate_fast_threshold
        
        if speech_rate > fast_threshold:
            scores['fast_speech_rate'] = min(1.0, (speech_rate - fast_threshold) / 4)
        else:
            scores['fast_speech_rate'] = 0.0
        
        # 4. 유성음 비율 (매우 낮은 비율 = 비명, 샤우팅)
        voiced_ratio = audio_features.get('voiced_unvoiced_ratio', {}).get('voiced_ratio', 0.5)
        
        # config에서 임계값 사용
        voiced_threshold = self.voiced_ratio_low_threshold
        
        if voiced_ratio < voiced_threshold:
            scores['low_voiced_ratio'] = 1.0 - voiced_ratio
        else:
            scores['low_voiced_ratio'] = 0.0
        
        # 5. 지터/시머 (매우 높은 값 = 떨림, 불안)
        jitter = audio_features.get('jitter_shimmer', {}).get('jitter', 0)
        shimmer = audio_features.get('jitter_shimmer', {}).get('shimmer', 0)
        
        # config에서 임계값 사용
        jitter_thresh = self.jitter_threshold
        shimmer_thresh = self.shimmer_threshold
        
        if jitter > jitter_thresh:
            scores['high_jitter'] = min(1.0, jitter / (jitter_thresh * 2))
        else:
            scores['high_jitter'] = max(0.0, jitter / jitter_thresh) * 0.3
        
        if shimmer > shimmer_thresh:
            scores['high_shimmer'] = min(1.0, shimmer / (shimmer_thresh * 2))
        else:
            scores['high_shimmer'] = max(0.0, shimmer / shimmer_thresh) * 0.3
        
        return scores
    
    def calculate_confidence_score(
        self, 
        audio_features: Dict[str, Any],
        llm_priority: str,
        llm_is_emergency: bool
    ) -> Dict[str, Any]:
        """
        음성 특성 + LLM 분석으로부터 최종 신뢰도 점수 계산
        
        Args:
            audio_features: 추출된 음성 특성
            llm_priority: LLM이 판정한 우선순위 (CRITICAL, HIGH, MEDIUM, LOW)
            llm_is_emergency: LLM이 판정한 긴급 여부
        
        Returns:
            신뢰도 정보 {
                'voice_emergency_score': 0-1 (음성 특성 기반 응급 가능성),
                'combined_priority': 최종 우선순위,
                'confidence': 0-1 (신뢰도),
                'breakdown': 상세 점수
            }
        """
        # 음성 특성으로부터 응급 신호 계산
        voice_indicators = self.analyze_emergency_indicators(audio_features)
        
        # 음성 기반 응급 점수
        voice_emergency_score = np.mean(list(voice_indicators.values())) if voice_indicators else 0.0
        
        # LLM 우선순위를 수치로 변환
        priority_weights = {
            'CRITICAL': 1.0,
            'HIGH': 0.75,
            'MEDIUM': 0.5,
            'LOW': 0.25
        }
        llm_score = priority_weights.get(llm_priority, 0.5)
        
        # config에서 가중치 사용
        final_score = (llm_score * self.llm_weight) + (voice_emergency_score * self.voice_weight)
        
        # config에서 우선순위 임계값 사용
        if final_score >= self.priority_critical:
            final_priority = 'CRITICAL'
        elif final_score >= self.priority_high:
            final_priority = 'HIGH'
        elif final_score >= self.priority_medium:
            final_priority = 'MEDIUM'
        else:
            final_priority = 'LOW'
        
        # 신뢰도: 음성 특성의 일관성
        # 여러 지표가 높으면 신뢰도 증가
        high_indicators = sum(1 for v in voice_indicators.values() if v > 0.5)
        confidence = high_indicators / len(voice_indicators) if voice_indicators else 0.5
        
        return {
            'voice_emergency_score': float(voice_emergency_score),
            'llm_score': float(llm_score),
            'combined_score': float(final_score),
            'final_priority': final_priority,
            'confidence': float(confidence),
            'breakdown': {
                'voice_indicators': voice_indicators,
                'llm_priority': llm_priority,
                'llm_is_emergency': llm_is_emergency
            }
        }


if __name__ == "__main__":
    print("=" * 70)
    print("🎤 음성 특성 분석 모듈 테스트")
    print("=" * 70)
    
    analyzer = VoiceCharacteristicsAnalyzer()
    
    # 테스트 오디오 파일이 있으면 분석
    print("\n📌 librosa 설치:")
    print("   pip install librosa")
    print("\n테스트를 위해 오디오 파일을 제공하세요.")
