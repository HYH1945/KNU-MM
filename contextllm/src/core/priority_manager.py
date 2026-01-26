#!/usr/bin/env python3
"""
우선순위 큐 시스템
"""

from enum import Enum
from datetime import datetime
from collections import deque
import json

class PriorityLevel(Enum):
    """우선순위 레벨"""
    CRITICAL = 1    # 긴급
    HIGH = 2        # 높음
    MEDIUM = 3      # 중간
    LOW = 4         # 낮음

class PriorityQueue:
    """우선순위 큐 관리"""
    
    def __init__(self, max_size=100):
        self.queues = {
            PriorityLevel.CRITICAL: deque(),
            PriorityLevel.HIGH: deque(),
            PriorityLevel.MEDIUM: deque(),
            PriorityLevel.LOW: deque(),
        }
        self.max_size = max_size
        self.total_items = 0
    
    def add_item(self, text, priority, analysis=None):
        """항목 추가"""
        if self.total_items >= self.max_size:
            # 가장 낮은 우선순위 항목 제거
            for level in [PriorityLevel.LOW, PriorityLevel.MEDIUM, PriorityLevel.HIGH]:
                if self.queues[level]:
                    self.queues[level].popleft()
                    self.total_items -= 1
                    break
        
        item = {
            'timestamp': datetime.now().isoformat(),
            'text': text,
            'analysis': analysis or {}
        }
        self.queues[priority].append(item)
        self.total_items += 1
        return item
    
    def get_next_item(self):
        """다음 우선순위 항목 가져오기"""
        for level in [PriorityLevel.CRITICAL, PriorityLevel.HIGH, PriorityLevel.MEDIUM, PriorityLevel.LOW]:
            if self.queues[level]:
                return self.queues[level].popleft()
        return None
    
    def get_stats(self):
        """통계"""
        return {
            'total': self.total_items,
            'critical': len(self.queues[PriorityLevel.CRITICAL]),
            'high': len(self.queues[PriorityLevel.HIGH]),
            'medium': len(self.queues[PriorityLevel.MEDIUM]),
            'low': len(self.queues[PriorityLevel.LOW]),
        }
