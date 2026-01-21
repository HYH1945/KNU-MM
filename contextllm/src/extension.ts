import * as vscode from 'vscode';
import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
// @ts-ignore
import * as recorder from 'node-record-lpcm16';
// @ts-ignore
import * as say from 'say';

let isRecording = false;
let recordingProcess: any = null;
let recordingTimer: NodeJS.Timeout | null = null;


export function activate(context: vscode.ExtensionContext) {
    // 프로젝트 루트 경로 (상대 경로)
    const projectRoot = context.extensionPath;
    const resultsDir = path.join(projectRoot, 'transcriptions');
    const pythonPath = path.join(projectRoot, '.venv', 'bin', 'python3');
    const pythonScriptPath = path.join(projectRoot, 'whisper_service.py');
    const tempAudioPath = path.join(projectRoot, 'recording.wav');

    // 결과 디렉토리 생성
    if (!fs.existsSync(resultsDir)) {
        fs.mkdirSync(resultsDir, { recursive: true });
    }

    console.log(`[Whisper Extension] 활성화됨 - Python 경로: ${pythonPath}`);

    let disposable = vscode.commands.registerCommand('whisper-tts.toggleMic', async () => {
        if (!isRecording) {
            isRecording = true;
            vscode.window.showInformationMessage('🎤 로컬 Whisper 녹음 시작... (10초 제한)');
            
            const fileStream = fs.createWriteStream(tempAudioPath);
            recordingProcess = recorder.record({
                sampleRate: 16000,
                recordProgram: 'rec',
            });
            recordingProcess.stream().pipe(fileStream);

            // 10초 후 자동 중지
            recordingTimer = setTimeout(() => {
                if (isRecording && recordingProcess) {
                    isRecording = false;
                    recordingProcess.stop();
                    vscode.window.showInformationMessage('⏱️ 시간 초과 - 녹음 자동 중지 및 분석 시작...');
                    performTranscription();
                }
            }, 10000);

        } else {
            // 수동 중지
            isRecording = false;
            if (recordingTimer) {
                clearTimeout(recordingTimer);
                recordingTimer = null;
            }
            if (recordingProcess) {
                recordingProcess.stop();
                vscode.window.showInformationMessage('⏸️ 녹음 중지 - 분석 중 (로컬 CPU/GPU 사용)...');
                performTranscription();
            }
        }
    });

    // 변환 실행 함수
    const performTranscription = () => {
        // --- 가상환경의 파이썬 실행 (절대 경로 + 환경변수) ---
        const pythonProcess = spawn(pythonPath, [pythonScriptPath, tempAudioPath], {
            env: {
                ...process.env,
                PYTHONUNBUFFERED: '1'
            }
        });

        let outputText = '';

        pythonProcess.stdout.on('data', (data) => {
            outputText += data.toString();
        });

        pythonProcess.on('close', (code) => {
            if (code === 0) {
                const resultText = outputText.trim();
                
                // 1. 타임스탬프와 함께 txt 파일에 저장
                saveToFile(resultText);
                
                // 2. 에디터에 텍스트 입력
                const editor = vscode.window.activeTextEditor;
                if (editor) {
                    editor.edit(editBuilder => {
                        editBuilder.insert(editor.selection.active, resultText + '\n');
                    });
                }

                // 3. TTS 읽어주기
                say.speak(resultText);
                vscode.window.showInformationMessage(`✅ 변환 완료! "${resultText.substring(0, 30)}..."`);
            } else {
                vscode.window.showErrorMessage(`❌ Whisper 오류 (코드: ${code})`);
            }
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error(`[Whisper Error] ${data}`);
        });
    };

    // 파일 저장 함수
    const saveToFile = (text: string) => {
        const timestamp = new Date().toISOString();
        const dateFolder = path.join(resultsDir, new Date().toISOString().split('T')[0]);
        
        // 날짜별 폴더 생성
        if (!fs.existsSync(dateFolder)) {
            fs.mkdirSync(dateFolder, { recursive: true });
        }

        // 1. TXT 파일 (누적 형식)
        const txtFile = path.join(dateFolder, 'transcriptions.txt');
        const txtContent = `[${timestamp}] ${text}\n`;
        fs.appendFileSync(txtFile, txtContent);

        // 2. JSON 파일 (LLM 호환 형식)
        const jsonFile = path.join(dateFolder, 'transcriptions.json');
        const jsonEntry = {
            timestamp,
            text,
            model: 'whisper-base',
            language: 'ko'
        };

        if (fs.existsSync(jsonFile)) {
            const existing = JSON.parse(fs.readFileSync(jsonFile, 'utf-8'));
            existing.push(jsonEntry);
            fs.writeFileSync(jsonFile, JSON.stringify(existing, null, 2));
        } else {
            fs.writeFileSync(jsonFile, JSON.stringify([jsonEntry], null, 2));
        }

        // 3. 개별 JSON 파일 (분석용)
        const individualJsonFile = path.join(dateFolder, `${timestamp.replace(/[:.]/g, '-')}.json`);
        fs.writeFileSync(individualJsonFile, JSON.stringify(jsonEntry, null, 2));

        console.log(`[Saved] TXT: ${txtFile}, JSON: ${jsonFile}`);
    };

    context.subscriptions.push(disposable);
}