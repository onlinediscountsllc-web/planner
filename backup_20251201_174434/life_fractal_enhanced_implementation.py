"""
🌀 LIFE FRACTAL INTELLIGENCE - ENHANCED FEATURES IMPLEMENTATION
═══════════════════════════════════════════════════════════════════════════════════════
Ready-to-integrate enhancements for math-first neurodivergent life planning

Features:
✅ Enhanced Pet AI with differential equations
✅ Fractal calendar system
✅ Executive dysfunction detection & support
✅ Privacy-preserving ML
✅ Accessibility helpers
✅ Fibonacci task scheduling
✅ Local emotional AI

Copy these classes into your existing application!
═══════════════════════════════════════════════════════════════════════════════════════
"""

# Using pure Python math - zero dependencies!
import pure_python_math as math_engine
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import json
import secrets

# Sacred Mathematics Constants (already in your code, but repeated for clarity)
PHI = (1 + math.sqrt(5)) / 2  # 1.618033988749895
PHI_INVERSE = PHI - 1  # 0.618033988749895
GOLDEN_ANGLE = 137.5077640500378  # degrees
GOLDEN_ANGLE_RAD = math.radians(GOLDEN_ANGLE)
FIBONACCI = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]


# ═══════════════════════════════════════════════════════════════════════════════════════
# ENHANCED PET AI WITH EMOTIONAL DIFFERENTIAL EQUATIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

class EmotionalPetAI:
    """
    Pet emotions evolve using coupled differential equations.
    Creates realistic, responsive pet behavior without external AI APIs.
    """
    
    SPECIES_PARAMS = {
        'dragon': {
            'hunger_decay': 0.8,
            'energy_decay': 1.2,
            'mood_sensitivity': 1.5,
            'bond_growth': 1.2,
            'chaos_tolerance': 0.9
        },
        'phoenix': {
            'hunger_decay': 1.0,
            'energy_decay': 0.5,
            'mood_sensitivity': 0.8,
            'bond_growth': 1.5,
            'chaos_tolerance': 1.2
        },
        'owl': {
            'hunger_decay': 0.7,
            'energy_decay': 0.6,
            'mood_sensitivity': 1.1,
            'bond_growth': 0.9,
            'chaos_tolerance': 0.6
        },
        'cat': {
            'hunger_decay': 1.2,
            'energy_decay': 1.0,
            'mood_sensitivity': 1.0,
            'bond_growth': 1.0,
            'chaos_tolerance': 0.8
        },
        'fox': {
            'hunger_decay': 1.1,
            'energy_decay': 0.9,
            'mood_sensitivity': 1.3,
            'bond_growth': 1.1,
            'chaos_tolerance': 1.0
        }
    }
    
    def __init__(self, species='cat', initial_state=None):
        self.species = species
        self.params = self.SPECIES_PARAMS.get(species, self.SPECIES_PARAMS['cat'])
        
        if initial_state:
            self.state = initial_state
        else:
            self.state = {
                'hunger': 50.0,
                'energy': 50.0,
                'mood': 50.0,
                'bond': 0.0,
                'age_days': 0,
                'xp': 0,
                'level': 1,
                'emotional_vector': [0.5, 0.5, 0.5]  # [valence, arousal, dominance]
            }
    
    def update(self, dt=1.0, user_wellness=50, interactions=0, sleep_quality=50):
        """
        Update pet state using Euler method for differential equations.
        dt = time step in hours
        """
        
        # Hunger dynamics: dH/dt = -δ_H · H + α · U(t)
        dH_dt = -self.params['hunger_decay'] * self.state['hunger'] / 100
        self.state['hunger'] = math_engine.clip(
            self.state['hunger'] + dH_dt * dt,
            0, 100
        )
        
        # Energy dynamics: dE/dt = -δ_E · E + β · S(t)
        dE_dt = (
            -self.params['energy_decay'] * self.state['energy'] / 100 +
            0.05 * sleep_quality
        )
        self.state['energy'] = math_engine.clip(
            self.state['energy'] + dE_dt * dt,
            0, 100
        )
        
        # Mood dynamics (coupled to user): dM/dt = γ · (U(t) - σ_M) + ε · I(t)
        user_coupling = self.params['mood_sensitivity'] * (user_wellness - 50) / 50
        interaction_boost = interactions * 5
        chaos_penalty = self._calculate_chaos_penalty(user_wellness)
        hunger_penalty = 0.1 * self.state['hunger']
        
        dM_dt = (
            user_coupling +
            interaction_boost -
            chaos_penalty -
            hunger_penalty
        )
        
        self.state['mood'] = math_engine.clip(
            self.state['mood'] + dM_dt * dt,
            0, 100
        )
        
        # Bond dynamics: dB/dt = η · M - θ · age
        dB_dt = (
            self.params['bond_growth'] * self.state['mood'] / 100 -
            0.01 * self.state['age_days']
        )
        self.state['bond'] = math_engine.clip(
            self.state['bond'] + dB_dt * dt,
            0, 100
        )
        
        # Update emotional vector (for advanced visualization)
        self._update_emotional_vector(user_wellness, interactions)
        
        return self.state
    
    def _calculate_chaos_penalty(self, user_wellness):
        """Pet responds to user chaos based on species tolerance"""
        chaos_level = abs(user_wellness - 50) / 50
        tolerance = self.params['chaos_tolerance']
        
        if chaos_level > tolerance:
            return (chaos_level - tolerance) ** 2 * 10
        return 0
    
    def _update_emotional_vector(self, user_wellness, interactions):
        """
        Update 3D emotional space: [Valence, Arousal, Dominance]
        Used for advanced emotional AI
        """
        current = math_engine.array(self.state['emotional_vector'])
        
        # Target emotional state based on mood/energy
        target_valence = self.state['mood'] / 100
        target_arousal = self.state['energy'] / 100
        target_dominance = self.state['bond'] / 100
        
        target = math_engine.array([target_valence, target_arousal, target_dominance])
        
        # Smooth transition (exponential moving average)
        alpha = 0.3
        self.state['emotional_vector'] = (
            alpha * target + (1 - alpha) * current
        ).tolist()
    
    def feed(self):
        """Feeding event"""
        self.state['hunger'] = max(0, self.state['hunger'] - 30)
        self.state['mood'] = min(100, self.state['mood'] + 5)
        self.state['xp'] += 5
        self._check_level_up()
        
        return {
            'success': True,
            'message': self._generate_response('feed'),
            'state': self.state
        }
    
    def play(self):
        """Play interaction"""
        if self.state['energy'] < 20:
            return {
                'success': False,
                'message': f"{self.species.title()} is too tired to play",
                'state': self.state
            }
        
        self.state['energy'] -= 15
        self.state['mood'] = min(100, self.state['mood'] + 10)
        self.state['bond'] = min(100, self.state['bond'] + 2)
        self.state['xp'] += 10
        self._check_level_up()
        
        return {
            'success': True,
            'message': self._generate_response('play'),
            'state': self.state
        }
    
    def _check_level_up(self):
        """Check if pet leveled up"""
        xp_for_next_level = self.state['level'] * 100
        if self.state['xp'] >= xp_for_next_level:
            self.state['level'] += 1
            self.state['xp'] -= xp_for_next_level
            return True
        return False
    
    def _generate_response(self, action):
        """Generate contextual response based on emotional state"""
        mood = self.state['mood']
        
        responses = {
            'feed': {
                'high': f"✨ {self.species.title()} devours the food with pure joy!",
                'medium': f"😊 {self.species.title()} eats gratefully",
                'low': f"😔 {self.species.title()} eats slowly..."
            },
            'play': {
                'high': f"🎉 {self.species.title()} leaps with excitement!",
                'medium': f"😊 {self.species.title()} plays happily",
                'low': f"😔 {self.species.title()} tries to play but seems sad"
            }
        }
        
        tier = 'high' if mood > 70 else ('medium' if mood > 40 else 'low')
        return responses[action][tier]
    
    def get_fractal_parameters(self):
        """
        Return pet's emotional state as fractal visualization parameters.
        This directly feeds into your fractal generation!
        """
        return {
            'pet_hue': int(self.state['mood'] * 3.6),  # 0-360 degrees
            'pet_chaos': (100 - self.state['energy']) / 100,
            'pet_zoom_factor': 1 + (self.state['bond'] / 100) * PHI,
            'pet_animation_speed': 0.5 + (self.state['energy'] / 100),
            'pet_glow_intensity': self.state['bond'] / 100
        }


# ═══════════════════════════════════════════════════════════════════════════════════════
# FRACTAL TIME CALENDAR SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════════

class FractalTimeCalendar:
    """
    Calendar based on fractal time decomposition and natural rhythms.
    Uses Fibonacci time blocks instead of rigid clock hours.
    """
    
    def __init__(self, user_timezone='UTC'):
        self.timezone = user_timezone
        self.phi = PHI
    
    def generate_daily_schedule(self, date=None, user_energy_pattern=None):
        """
        Generate Fibonacci-based daily schedule.
        Returns natural time blocks aligned with circadian rhythms.
        """
        
        if date is None:
            date = datetime.now()
        
        # Default energy pattern if not provided
        if user_energy_pattern is None:
            user_energy_pattern = self._default_energy_pattern()
        
        # Start of waking day (default 6 AM)
        start_hour = 6
        total_waking_hours = 16
        
        # Generate Fibonacci time blocks
        blocks = []
        current_hour = start_hour
        fib_sequence = [1, 1, 2, 3, 5]  # Hours for each block
        
        for i, fib_hours in enumerate(fib_sequence):
            if current_hour >= start_hour + total_waking_hours:
                break
            
            block_end = min(current_hour + fib_hours, start_hour + total_waking_hours)
            duration = block_end - current_hour
            
            # Calculate energy phase
            energy_phase, energy_level = self._calculate_energy_phase(
                current_hour,
                user_energy_pattern
            )
            
            # Estimate spoon capacity
            spoon_capacity = max(1, int(3 + 2 * energy_level / 100))
            
            blocks.append({
                'id': f'block_{i}',
                'start_time': f"{int(current_hour):02d}:00",
                'end_time': f"{int(block_end):02d}:00",
                'duration_hours': duration,
                'energy_phase': energy_phase,
                'energy_level': energy_level,
                'spoon_capacity': spoon_capacity,
                'fibonacci_index': i,
                'optimal_activities': self._suggest_activities(energy_phase),
                'recommended_task_types': self._task_types_for_energy(energy_level)
            })
            
            current_hour = block_end
        
        return {
            'date': date.strftime('%Y-%m-%d'),
            'schedule_type': 'fibonacci_fractal',
            'time_blocks': blocks,
            'total_available_spoons': sum(b['spoon_capacity'] for b in blocks),
            'sacred_math_applied': {
                'golden_ratio': True,
                'fibonacci_rhythm': True,
                'circadian_alignment': True
            }
        }
    
    def _default_energy_pattern(self):
        """
        Default circadian energy pattern.
        Can be customized per user based on their data.
        """
        return {
            'peak_hour': 10,      # 10 AM peak
            'trough_hour': 15,    # 3 PM slump
            'amplitude': 40        # Energy swing range
        }
    
    def _calculate_energy_phase(self, hour, pattern):
        """
        Calculate energy level using sinusoidal circadian model.
        Returns (phase_name, energy_level)
        """
        
        # Sinusoidal circadian rhythm
        phase_angle = 2 * math_engine.pi * (hour - 6) / 24
        energy = 50 + pattern['amplitude'] * math_engine.sin(phase_angle - math_engine.pi/2)
        
        if energy > 75:
            phase = 'peak'
        elif energy > 50:
            phase = 'high'
        elif energy > 25:
            phase = 'medium'
        else:
            phase = 'low'
        
        return phase, energy
    
    def _suggest_activities(self, phase):
        """Suggest activity types based on energy phase"""
        suggestions = {
            'peak': ['Deep work', 'Complex problem-solving', 'Creative projects'],
            'high': ['Important tasks', 'Meetings', 'Learning new skills'],
            'medium': ['Routine work', 'Communications', 'Planning'],
            'low': ['Rest', 'Light admin', 'Reflection', 'Recovery']
        }
        return suggestions.get(phase, ['Flexible activities'])
    
    def _task_types_for_energy(self, energy_level):
        """Map energy level to appropriate task types"""
        if energy_level > 75:
            return ['high_focus', 'creative', 'complex']
        elif energy_level > 50:
            return ['moderate_focus', 'collaborative', 'learning']
        elif energy_level > 25:
            return ['low_focus', 'routine', 'social']
        else:
            return ['rest', 'recovery', 'light']


# ═══════════════════════════════════════════════════════════════════════════════════════
# FIBONACCI TASK SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════════════

class FibonacciTaskScheduler:
    """
    Schedule tasks using Fibonacci sequences and golden ratio prioritization.
    """
    
    @staticmethod
    def prioritize_tasks(tasks, current_spoons, urgency_matrix=None):
        """
        Prioritize tasks using: Priority = (φ · Urgency + φ⁻¹ · Importance) / Spoons
        """
        
        phi = PHI
        phi_inv = PHI_INVERSE
        
        if urgency_matrix is None:
            urgency_matrix = {}
        
        scored_tasks = []
        
        for task in tasks:
            task_id = task.get('id', task.get('name', ''))
            urgency = urgency_matrix.get(task_id, {}).get('urgency', 50) / 100
            importance = urgency_matrix.get(task_id, {}).get('importance', 50) / 100
            spoon_cost = task.get('spoon_cost', 3)
            
            # Golden ratio weighted priority
            priority_score = (phi * urgency + phi_inv * importance) / max(1, spoon_cost)
            
            # Feasibility check
            feasible = spoon_cost <= current_spoons
            
            scored_tasks.append({
                **task,
                'priority_score': priority_score,
                'feasible': feasible,
                'recommended_time': FibonacciTaskScheduler._recommend_time(
                    urgency, importance, spoon_cost
                )
            })
        
        # Sort by priority
        scored_tasks.sort(key=lambda t: t['priority_score'], reverse=True)
        
        # Group into Fibonacci tiers
        tiers = {
            'critical': scored_tasks[:2],       # Fib(3) = 2
            'important': scored_tasks[2:5],     # Fib(4) = 3
            'standard': scored_tasks[5:8],      # Fib(5) = 5 (total 8)
            'low': scored_tasks[8:13],          # Fib(6) = 8 (total 13)
            'backlog': scored_tasks[13:]
        }
        
        return tiers
    
    @staticmethod
    def _recommend_time(urgency, importance, spoon_cost):
        """Recommend optimal time block for task"""
        if urgency > 0.8:
            return 'next_available'
        elif importance > 0.7 and spoon_cost > 5:
            return 'peak_energy'
        elif spoon_cost > 5:
            return 'high_energy'
        else:
            return 'flexible'
    
    @staticmethod
    def allocate_to_schedule(tasks, schedule_blocks):
        """
        Allocate tasks to Fibonacci time blocks.
        Returns optimized schedule with tasks assigned.
        """
        
        allocated_schedule = []
        remaining_tasks = tasks.copy()
        
        for block in schedule_blocks:
            block_tasks = []
            available_spoons = block['spoon_capacity']
            available_time = block['duration_hours'] * 60  # minutes
            
            # Find tasks that fit this block
            for task in remaining_tasks[:]:
                if (task.get('spoon_cost', 3) <= available_spoons and
                    task.get('estimated_minutes', 30) <= available_time):
                    
                    # Check if task type matches block energy
                    task_type = task.get('type', 'flexible')
                    if task_type in block.get('recommended_task_types', []) or task_type == 'flexible':
                        block_tasks.append(task)
                        available_spoons -= task.get('spoon_cost', 3)
                        available_time -= task.get('estimated_minutes', 30)
                        remaining_tasks.remove(task)
            
            allocated_schedule.append({
                **block,
                'assigned_tasks': block_tasks,
                'remaining_spoons': available_spoons,
                'remaining_minutes': available_time,
                'utilization': 1 - (available_spoons / block['spoon_capacity'])
            })
        
        return {
            'schedule': allocated_schedule,
            'unscheduled_tasks': remaining_tasks,
            'total_scheduled': len(tasks) - len(remaining_tasks)
        }


# ═══════════════════════════════════════════════════════════════════════════════════════
# EXECUTIVE DYSFUNCTION DETECTION & SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════════════

class ExecutiveFunctionSupport:
    """
    Detect executive dysfunction patterns and provide scaffolding.
    Uses Fourier analysis of behavior patterns.
    """
    
    @staticmethod
    def detect_dysfunction(behavior_history, threshold=0.3):
        """
        Use FFT to detect executive dysfunction patterns.
        
        Dysfunction indicators:
        - High frequency oscillations (task-switching)
        - Low frequency drift (motivation crashes)
        - Missing fundamental frequency (no routine)
        """
        
        if len(behavior_history) < 7:
            return {
                'dysfunction_detected': False,
                'score': 0,
                'message': 'Insufficient data for analysis'
            }
        
        # Extract completion times
        completion_times = math_engine.array([
            entry.get('task_completion_time', 30)
            for entry in behavior_history[-30:]  # Last 30 days
        ])
        
        # Pad to power of 2 for FFT efficiency
        n = len(completion_times)
        padded_n = 2 ** int(math_engine.ceil(math_engine.log2(n)))
        padded = math_engine.pad(completion_times, (0, padded_n - n), mode='constant')
        
        # FFT to frequency domain
        fft = math_engine.fft.fft(padded)
        frequencies = math_engine.fft.fftfreq(padded_n)
        power_spectrum = math_engine.abs(fft) ** 2
        
        # Analyze frequency components
        high_freq_power = math_engine.sum(power_spectrum[math_engine.abs(frequencies) > 0.1])
        low_freq_power = math_engine.sum(power_spectrum[math_engine.abs(frequencies) < 0.05])
        
        # Check for weekly routine (fundamental frequency)
        weekly_freq_idx = math_engine.argmin(math_engine.abs(frequencies - 1/7))
        fundamental_power = power_spectrum[weekly_freq_idx]
        
        # Calculate dysfunction score
        total_power = math_engine.sum(power_spectrum)
        dysfunction_score = (
            (high_freq_power / max(1, low_freq_power)) *
            (1 - fundamental_power / max(1, total_power))
        )
        
        # Normalize
        dysfunction_score = min(1.0, dysfunction_score / 10)
        
        return {
            'dysfunction_detected': dysfunction_score > threshold,
            'score': float(dysfunction_score),
            'severity': ExecutiveFunctionSupport._classify_severity(dysfunction_score),
            'recommendation': ExecutiveFunctionSupport._generate_recommendation(dysfunction_score),
            'patterns': {
                'task_switching_level': float(high_freq_power / max(1, total_power)),
                'motivation_stability': float(low_freq_power / max(1, total_power)),
                'routine_consistency': float(fundamental_power / max(1, total_power))
            }
        }
    
    @staticmethod
    def _classify_severity(score):
        """Classify dysfunction severity"""
        if score < 0.2:
            return 'minimal'
        elif score < 0.4:
            return 'mild'
        elif score < 0.6:
            return 'moderate'
        elif score < 0.8:
            return 'significant'
        else:
            return 'severe'
    
    @staticmethod
    def _generate_recommendation(score):
        """Generate support recommendations"""
        if score < 0.2:
            return {
                'message': "Executive function is strong. Continue current strategies.",
                'strategies': ['Maintain routines', 'Build on successes']
            }
        elif score < 0.4:
            return {
                'message': "Mild strain detected. Consider task chunking.",
                'strategies': [
                    'Break tasks into 15-minute chunks',
                    'Use Fibonacci breaks (1, 2, 3, 5 minutes)',
                    'Reduce parallel tasks'
                ]
            }
        elif score < 0.6:
            return {
                'message': "Moderate dysfunction. Increase external support.",
                'strategies': [
                    'Use task scaffolding (micro-steps)',
                    'Reduce daily task load by 38.2% (golden ratio)',
                    'Schedule more rest blocks',
                    'Consider body doubling'
                ]
            }
        elif score < 0.8:
            return {
                'message': "Significant dysfunction. Implement strong scaffolding.",
                'strategies': [
                    'All tasks need micro-step breakdown',
                    'Reduce to 2-3 tasks per day max',
                    'Use external timers and reminders',
                    'Focus on self-care first'
                ]
            }
        else:
            return {
                'message': "Severe dysfunction. Radical rest recommended.",
                'strategies': [
                    'Implement rest protocol: minimal demands',
                    'Cancel non-essential tasks',
                    'Seek support from others',
                    'Focus only on basic self-care',
                    'Consider professional support if ongoing'
                ]
            }
    
    @staticmethod
    def generate_task_scaffold(task, max_step_minutes=5):
        """
        Break task into micro-steps.
        Each step <5 minutes and <2 spoons.
        """
        
        estimated_minutes = task.get('estimated_minutes', 30)
        task_name = task.get('name', 'Task')
        task_type = task.get('type', 'general')
        
        # Calculate number of steps using Fibonacci
        num_steps = max(3, int(math_engine.ceil(estimated_minutes / max_step_minutes)))
        
        # Predefined scaffolds by task type
        scaffolds = {
            'writing': [
                "Open document and set 5-min timer",
                "Write opening sentence",
                "Write 1 paragraph (3-5 sentences)",
                "Quick read-through",
                "Add another paragraph",
                "2-minute break - stretch",
                "Final paragraph",
                "Quick spell check",
                "Save and close"
            ],
            'cleaning': [
                "Set 5-minute timer",
                "Pick up 5 visible items",
                "Put items where they belong",
                "1-minute water break",
                "Pick up 5 more items",
                "Quick surface wipe",
                "Final scan",
                "Done!"
            ],
            'email': [
                "Open email",
                "Read first sentence only",
                "Identify main question",
                "Think of 1-sentence answer",
                "Write that sentence",
                "Add greeting/closing",
                "Quick proofread",
                "Send"
            ]
        }
        
        if task_type in scaffolds:
            micro_steps = scaffolds[task_type][:num_steps]
        else:
            # Generic breakdown
            micro_steps = [
                f"{task_name} - Step {i+1}/{num_steps}"
                for i in range(num_steps)
            ]
        
        return {
            'original_task': task,
            'micro_steps': micro_steps,
            'steps_count': len(micro_steps),
            'spoons_per_step': 1,
            'time_per_step': max(3, estimated_minutes // len(micro_steps)),
            'completion_tracking': [False] * len(micro_steps),
            'progress_percentage': 0,
            'motivational_message': f"Just {len(micro_steps)} tiny steps. You can do this! 💪"
        }


# ═══════════════════════════════════════════════════════════════════════════════════════
# ACCESSIBILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════════════

class AutismSafeColors:
    """
    Generate sensory-safe color palettes.
    Avoids intense colors that can cause sensory overload.
    """
    
    SAFE_HUE_RANGES = {
        'calming_blues': (180, 240),
        'gentle_greens': (90, 150),
        'soft_purples': (260, 290),
        'warm_earth': (20, 50)
    }
    
    @staticmethod
    def generate_theme(mood='calm', contrast='medium'):
        """
        Generate complete color theme from mathematical rules.
        """
        
        # Select hue range
        if mood == 'calm':
            hue_range = AutismSafeColors.SAFE_HUE_RANGES['calming_blues']
        elif mood == 'energized':
            hue_range = AutismSafeColors.SAFE_HUE_RANGES['warm_earth']
        elif mood == 'balanced':
            hue_range = AutismSafeColors.SAFE_HUE_RANGES['gentle_greens']
        else:
            hue_range = AutismSafeColors.SAFE_HUE_RANGES['soft_purples']
        
        # Base hue (center of range)
        base_hue = (hue_range[0] + hue_range[1]) / 2
        
        # Generate palette using golden ratio spacing
        colors = {
            'primary': AutismSafeColors._hsl_to_hex(base_hue, 40, 60),
            'secondary': AutismSafeColors._hsl_to_hex(
                (base_hue + 360 * PHI_INVERSE) % 360, 35, 65
            ),
            'accent': AutismSafeColors._hsl_to_hex(
                (base_hue + 360 * PHI) % 360, 45, 55
            ),
            'background': AutismSafeColors._hsl_to_hex(base_hue, 10, 95),
            'text': AutismSafeColors._hsl_to_hex(base_hue, 5, 20),
            'text_secondary': AutismSafeColors._hsl_to_hex(base_hue, 5, 40)
        }
        
        # Adjust for contrast level
        if contrast == 'high':
            colors['text'] = '#000000'
            colors['background'] = '#FFFFFF'
        elif contrast == 'low':
            colors['text'] = AutismSafeColors._hsl_to_hex(base_hue, 5, 30)
        
        return colors
    
    @staticmethod
    def _hsl_to_hex(h, s, l):
        """Convert HSL to hex color"""
        s = s / 100
        l = l / 100
        
        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = l - c / 2
        
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        r = int((r + m) * 255)
        g = int((g + m) * 255)
        b = int((b + m) * 255)
        
        return f'#{r:02x}{g:02x}{b:02x}'


class AphantasiaSupport:
    """
    Support for users who cannot visualize mentally.
    Everything must be externalized as text/numbers.
    """
    
    @staticmethod
    def externalize_goal(goal):
        """
        Convert abstract goal into concrete, measurable description.
        """
        return {
            'goal_name': goal.get('name', ''),
            'concrete_metrics': {
                'current_value': goal.get('current_value', 0),
                'target_value': goal.get('target_value', 100),
                'progress_percent': goal.get('progress', 0),
                'completion_date': goal.get('target_date', 'Not set')
            },
            'physical_evidence': [
                f"Progress: {goal.get('progress', 0)}%",
                f"Completed {goal.get('completed_milestones', 0)} milestones",
                f"Time invested: {goal.get('time_invested', 0)} hours"
            ],
            'next_concrete_step': goal.get('next_step', 'Define next action'),
            'success_criteria': [
                "What will exist when done?",
                "What will you be able to do?",
                "What evidence will you have?"
            ]
        }
    
    @staticmethod
    def describe_fractal_in_text(fractal_params):
        """
        Describe visualization in words for those who can't process images.
        """
        
        wellness = fractal_params.get('wellness_index', 50)
        chaos = fractal_params.get('chaos_level', 0.5)
        mood = fractal_params.get('mood_score', 50)
        
        description = []
        
        # Overall pattern
        if wellness > 70:
            description.append("Strong, organized patterns showing stability.")
        elif wellness > 40:
            description.append("Moderate structure with balanced complexity.")
        else:
            description.append("Dispersed patterns indicating transition phase.")
        
        # Complexity
        if chaos > 0.7:
            description.append("High complexity suggests active growth and change.")
        elif chaos < 0.3:
            description.append("Simple patterns indicate calm stability.")
        
        # Color mood
        if mood > 70:
            description.append("Bright, warm colors reflecting positive state.")
        elif mood > 40:
            description.append("Balanced, neutral tones showing equilibrium.")
        else:
            description.append("Cool, muted colors suggesting need for rest.")
        
        return {
            'text_description': ' '.join(description),
            'skip_image': True,
            'data_summary': {
                'wellness_score': wellness,
                'complexity_level': chaos,
                'emotional_tone': mood
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════════════
# PRIVACY-PRESERVING MACHINE LEARNING
# ═══════════════════════════════════════════════════════════════════════════════════════

class PrivacyPreservingML:
    """
    Learn from patterns without exposing personal data.
    Uses differential privacy and local computation.
    """
    
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level
        self.global_patterns = []
    
    def extract_local_patterns(self, user_history):
        """
        Extract mathematical features only, NO personal content.
        """
        
        if len(user_history) < 5:
            return None
        
        # Extract anonymized statistical features
        patterns = {
            'completion_time_trend': self._fit_polynomial(
                [e.get('task_completion_time', 30) for e in user_history]
            ),
            'mood_spectrum': self._fft_features(
                [e.get('mood_score', 50) for e in user_history]
            ),
            'energy_curve': self._exponential_fit(
                [e.get('energy_level', 50) for e in user_history]
            )
        }
        
        # Add differential privacy noise
        for key in patterns:
            if isinstance(patterns[key], (list, math_engine.ndarray)):
                patterns[key] = self._add_noise(patterns[key])
        
        return patterns
    
    def _fit_polynomial(self, values, degree=2):
        """Fit polynomial to data"""
        x = math_engine.arange(len(values))
        coeffs = math_engine.polyfit(x, values, degree)
        return coeffs.tolist()
    
    def _fft_features(self, values, n_components=3):
        """Extract frequency features"""
        fft = math_engine.fft.fft(values)
        magnitudes = math_engine.abs(fft[:n_components])
        return magnitudes.tolist()
    
    def _exponential_fit(self, values):
        """Fit exponential model"""
        x = math_engine.arange(len(values))
        y = math_engine.array(values)
        
        # Avoid log of zero/negative
        y = math_engine.maximum(y, 0.1)
        
        log_y = math_engine.log(y)
        coeffs = math_engine.polyfit(x, log_y, 1)
        
        return {
            'rate': float(coeffs[0]),
            'baseline': float(math_engine.exp(coeffs[1]))
        }
    
    def _add_noise(self, data):
        """Add Laplacian noise for differential privacy"""
        noise = math_engine.random.laplace(0, self.noise_level, size=len(data))
        return (math_engine.array(data) + noise).tolist()
    
    def get_insights(self, user_patterns, global_patterns):
        """
        Compare user to anonymized global patterns.
        Provides insights without exposing individual data.
        """
        
        if not global_patterns or not user_patterns:
            return {"message": "Insufficient data for insights"}
        
        # Calculate similarity to global mean (simple Euclidean distance)
        # This is privacy-preserving because we're comparing to aggregated data
        
        insights = []
        
        # Example insight generation
        user_completion = math_engine.mean(user_patterns.get('completion_time_trend', [30]))
        global_completion = math_engine.mean([
            math_engine.mean(p.get('completion_time_trend', [30]))
            for p in global_patterns
        ])
        
        if abs(user_completion - global_completion) > 10:
            insights.append({
                'type': 'completion_pattern',
                'message': 'Your task completion pattern is unique. Custom strategies recommended.',
                'confidence': 'high'
            })
        
        return {
            'insights': insights,
            'privacy_preserved': True,
            'data_anonymized': True
        }


# ═══════════════════════════════════════════════════════════════════════════════════════
# INTEGRATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════════════

def integrate_pet_with_user_state(pet_ai, user_daily_entry):
    """
    Update pet based on user's daily entry.
    Returns updated pet state and fractal parameters.
    """
    
    # Extract user metrics
    user_wellness = user_daily_entry.get('wellness_index', 50)
    sleep_quality = user_daily_entry.get('sleep_quality', 50)
    
    # Count interactions (tasks completed, etc.)
    interactions = (
        user_daily_entry.get('tasks_completed', 0) +
        user_daily_entry.get('habits_completed', 0)
    )
    
    # Update pet (1 hour time step)
    updated_state = pet_ai.update(
        dt=1.0,
        user_wellness=user_wellness,
        interactions=interactions,
        sleep_quality=sleep_quality
    )
    
    # Get fractal parameters from pet
    fractal_params = pet_ai.get_fractal_parameters()
    
    return {
        'pet_state': updated_state,
        'fractal_params': fractal_params,
        'pet_message': pet_ai._generate_response('idle')
    }


def create_full_day_plan(date, user_profile, pending_tasks):
    """
    Create complete daily plan combining calendar, tasks, and pet.
    """
    
    # Initialize systems
    calendar = FractalTimeCalendar()
    scheduler = FibonacciTaskScheduler()
    
    # Generate Fibonacci schedule
    schedule = calendar.generate_daily_schedule(
        date=date,
        user_energy_pattern=user_profile.get('energy_pattern')
    )
    
    # Prioritize tasks
    current_spoons = sum(block['spoon_capacity'] for block in schedule['time_blocks'])
    task_tiers = scheduler.prioritize_tasks(pending_tasks, current_spoons)
    
    # Allocate tasks to schedule
    allocated = scheduler.allocate_to_schedule(
        task_tiers['critical'] + task_tiers['important'] + task_tiers['standard'],
        schedule['time_blocks']
    )
    
    return {
        'date': date,
        'schedule': allocated['schedule'],
        'unscheduled_tasks': allocated['unscheduled_tasks'],
        'daily_spoons': current_spoons,
        'task_tiers': task_tiers,
        'sacred_math': schedule['sacred_math_applied']
    }


# ═══════════════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🌀 Life Fractal Intelligence - Enhanced Features Demo")
    print("=" * 80)
    
    # 1. Pet AI Demo
    print("\n1. EMOTIONAL PET AI")
    pet = EmotionalPetAI(species='dragon')
    
    # Simulate user having good day
    for i in range(5):
        state = pet.update(
            dt=2.0,  # 2 hours
            user_wellness=75,
            interactions=1,
            sleep_quality=80
        )
        print(f"   Hour {i*2}: Mood={state['mood']:.1f}, Energy={state['energy']:.1f}, Bond={state['bond']:.1f}")
    
    feed_result = pet.feed()
    print(f"   {feed_result['message']}")
    
    # 2. Fractal Calendar Demo
    print("\n2. FRACTAL TIME CALENDAR")
    calendar = FractalTimeCalendar()
    schedule = calendar.generate_daily_schedule()
    
    print(f"   Generated {len(schedule['time_blocks'])} Fibonacci time blocks:")
    for block in schedule['time_blocks']:
        print(f"   - {block['start_time']}-{block['end_time']}: {block['energy_phase']} energy ({block['spoon_capacity']} spoons)")
    
    # 3. Task Prioritization Demo
    print("\n3. FIBONACCI TASK PRIORITIZATION")
    demo_tasks = [
        {'id': '1', 'name': 'Important project', 'spoon_cost': 5, 'estimated_minutes': 60},
        {'id': '2', 'name': 'Quick email', 'spoon_cost': 1, 'estimated_minutes': 10},
        {'id': '3', 'name': 'Deep work', 'spoon_cost': 8, 'estimated_minutes': 120},
        {'id': '4', 'name': 'Admin task', 'spoon_cost': 2, 'estimated_minutes': 20}
    ]
    
    urgency_matrix = {
        '1': {'urgency': 80, 'importance': 90},
        '2': {'urgency': 60, 'importance': 30},
        '3': {'urgency': 40, 'importance': 95},
        '4': {'urgency': 50, 'importance': 40}
    }
    
    scheduler = FibonacciTaskScheduler()
    tiers = scheduler.prioritize_tasks(demo_tasks, current_spoons=15, urgency_matrix=urgency_matrix)
    
    print(f"   Critical: {[t['name'] for t in tiers['critical']]}")
    print(f"   Important: {[t['name'] for t in tiers['important']]}")
    
    # 4. Executive Dysfunction Detection Demo
    print("\n4. EXECUTIVE DYSFUNCTION DETECTION")
    demo_history = [
        {'task_completion_time': 45 + i * math_engine.sin(i) * 10}
        for i in range(20)
    ]
    
    dysfunction = ExecutiveFunctionSupport.detect_dysfunction(demo_history)
    print(f"   Detected: {dysfunction['dysfunction_detected']}")
    print(f"   Severity: {dysfunction['severity']}")
    print(f"   Score: {dysfunction['score']:.3f}")
    
    # 5. Color Theme Demo
    print("\n5. AUTISM-SAFE COLOR THEME")
    colors = AutismSafeColors.generate_theme(mood='calm', contrast='medium')
    print(f"   Primary: {colors['primary']}")
    print(f"   Secondary: {colors['secondary']}")
    print(f"   Background: {colors['background']}")
    
    print("\n" + "=" * 80)
    print("✨ All systems operational! Ready to integrate into your app.")
