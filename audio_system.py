"""
COVER FACE - Audio System
==========================
Binaural beats, white/colored noise, ambient soundscapes
All generated in real-time using Web Audio API
"""

AUDIO_SYSTEM_JS = '''
// ============================================
// COVER FACE AUDIO SYSTEM
// ============================================

class CoverFaceAudio {
    constructor() {
        this.audioContext = null;
        this.masterGain = null;
        this.layers = {
            binaural: null,
            ambient: null,
            nature: null,
            music: null,
            noise: null
        };
        this.isPlaying = false;
        this.currentMood = 'calm';
        this.spatialSources = [];
    }
    
    async init() {
        try {
            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Create master gain
            this.masterGain = this.audioContext.createGain();
            this.masterGain.gain.value = 0.5;
            this.masterGain.connect(this.audioContext.destination);
            
            console.log('Audio system initialized');
            return true;
        } catch (error) {
            console.error('Audio initialization failed:', error);
            return false;
        }
    }
    
    // ============================================
    // BINAURAL BEATS GENERATOR
    // ============================================
    
    createBinauralBeat(baseFreq = 432, beatFreq = 10) {
        /*
        Binaural beats for brainwave entrainment:
        - Delta (0.5-4 Hz): Deep sleep
        - Theta (4-8 Hz): Meditation
        - Alpha (8-14 Hz): Relaxation
        - Beta (14-30 Hz): Focus
        - Gamma (30-100 Hz): Peak awareness
        */
        
        // Left ear
        const leftOsc = this.audioContext.createOscillator();
        leftOsc.frequency.value = baseFreq;
        
        // Right ear (slightly different frequency creates the "beat")
        const rightOsc = this.audioContext.createOscillator();
        rightOsc.frequency.value = baseFreq + beatFreq;
        
        // Create stereo panner
        const leftPanner = this.audioContext.createStereoPanner();
        leftPanner.pan.value = -1; // Full left
        
        const rightPanner = this.audioContext.createStereoPanner();
        rightPanner.pan.value = 1; // Full right
        
        // Create gain for volume control
        const gain = this.audioContext.createGain();
        gain.gain.value = 0.3;
        
        // Connect left channel
        leftOsc.connect(leftPanner);
        leftPanner.connect(gain);
        
        // Connect right channel
        rightOsc.connect(rightPanner);
        rightPanner.connect(gain);
        
        // Connect to master
        gain.connect(this.masterGain);
        
        return {
            left: leftOsc,
            right: rightOsc,
            gain: gain,
            start: () => {
                leftOsc.start();
                rightOsc.start();
            },
            stop: () => {
                leftOsc.stop();
                rightOsc.stop();
            }
        };
    }
    
    // ============================================
    // COLORED NOISE GENERATORS
    // ============================================
    
    createWhiteNoise() {
        // White noise: Equal energy across all frequencies
        const bufferSize = 2 * this.audioContext.sampleRate;
        const noiseBuffer = this.audioContext.createBuffer(1, bufferSize, this.audioContext.sampleRate);
        const output = noiseBuffer.getChannelData(0);
        
        for (let i = 0; i < bufferSize; i++) {
            output[i] = Math.random() * 2 - 1;
        }
        
        const noise = this.audioContext.createBufferSource();
        noise.buffer = noiseBuffer;
        noise.loop = true;
        
        return noise;
    }
    
    createPinkNoise() {
        // Pink noise: Equal energy per octave (1/f noise)
        const bufferSize = 2 * this.audioContext.sampleRate;
        const noiseBuffer = this.audioContext.createBuffer(1, bufferSize, this.audioContext.sampleRate);
        const output = noiseBuffer.getChannelData(0);
        
        let b0, b1, b2, b3, b4, b5, b6;
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0;
        
        for (let i = 0; i < bufferSize; i++) {
            const white = Math.random() * 2 - 1;
            b0 = 0.99886 * b0 + white * 0.0555179;
            b1 = 0.99332 * b1 + white * 0.0750759;
            b2 = 0.96900 * b2 + white * 0.1538520;
            b3 = 0.86650 * b3 + white * 0.3104856;
            b4 = 0.55000 * b4 + white * 0.5329522;
            b5 = -0.7616 * b5 - white * 0.0168980;
            output[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362;
            output[i] *= 0.11; // Normalize
            b6 = white * 0.115926;
        }
        
        const noise = this.audioContext.createBufferSource();
        noise.buffer = noiseBuffer;
        noise.loop = true;
        
        return noise;
    }
    
    createBrownNoise() {
        // Brown noise: Even more bass-heavy (1/f² noise)
        const bufferSize = 2 * this.audioContext.sampleRate;
        const noiseBuffer = this.audioContext.createBuffer(1, bufferSize, this.audioContext.sampleRate);
        const output = noiseBuffer.getChannelData(0);
        
        let lastOut = 0.0;
        for (let i = 0; i < bufferSize; i++) {
            const white = Math.random() * 2 - 1;
            output[i] = (lastOut + (0.02 * white)) / 1.02;
            lastOut = output[i];
            output[i] *= 3.5; // Normalize
        }
        
        const noise = this.audioContext.createBufferSource();
        noise.buffer = noiseBuffer;
        noise.loop = true;
        
        return noise;
    }
    
    // ============================================
    // AMBIENT SOUNDSCAPES
    // ============================================
    
    createAmbientDrone(frequency = 432) {
        // Create a slowly evolving drone using multiple oscillators
        const oscillators = [];
        const gains = [];
        const masterGain = this.audioContext.createGain();
        masterGain.gain.value = 0.2;
        
        // Create 5 oscillators at harmonic intervals
        for (let i = 0; i < 5; i++) {
            const osc = this.audioContext.createOscillator();
            const gain = this.audioContext.createGain();
            
            // Use Fibonacci ratios for harmonic spacing
            const fibonacci = [1, 1, 2, 3, 5];
            osc.frequency.value = frequency * fibonacci[i];
            osc.type = 'sine';
            
            // Slowly modulate gain
            gain.gain.setValueAtTime(0.1 / (i + 1), this.audioContext.currentTime);
            
            osc.connect(gain);
            gain.connect(masterGain);
            
            oscillators.push(osc);
            gains.push(gain);
        }
        
        masterGain.connect(this.masterGain);
        
        return {
            oscillators: oscillators,
            gains: gains,
            masterGain: masterGain,
            start: () => oscillators.forEach(osc => osc.start()),
            stop: () => oscillators.forEach(osc => osc.stop())
        };
    }
    
    // ============================================
    // MOOD-BASED PRESETS
    // ============================================
    
    getMoodPreset(mood) {
        const presets = {
            'stressed': {
                binaural: { base: 174, beat: 4 }, // Deep relaxation
                noise: 'pink',
                ambient: 174,
                nature: 'ocean'
            },
            'energized': {
                binaural: { base: 528, beat: 20 }, // Beta waves
                noise: 'white',
                ambient: 528,
                nature: 'forest'
            },
            'calm': {
                binaural: { base: 432, beat: 8 }, // Alpha waves
                noise: 'pink',
                ambient: 432,
                nature: 'rain'
            },
            'creative': {
                binaural: { base: 963, beat: 10 }, // Theta waves
                noise: 'brown',
                ambient: 963,
                nature: 'wind'
            },
            'focused': {
                binaural: { base: 852, beat: 40 }, // Gamma waves
                noise: 'pink',
                ambient: 852,
                nature: 'minimal'
            },
            'sleep': {
                binaural: { base: 432, beat: 2 }, // Delta waves
                noise: 'brown',
                ambient: 432,
                nature: 'night'
            }
        };
        
        return presets[mood] || presets['calm'];
    }
    
    // ============================================
    // PLAYBACK CONTROL
    // ============================================
    
    async start(mood = 'calm') {
        if (!this.audioContext) {
            await this.init();
        }
        
        // Resume context if needed
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
        
        this.currentMood = mood;
        const preset = this.getMoodPreset(mood);
        
        // Start binaural beats
        this.layers.binaural = this.createBinauralBeat(
            preset.binaural.base,
            preset.binaural.beat
        );
        this.layers.binaural.start();
        
        // Start ambient drone
        this.layers.ambient = this.createAmbientDrone(preset.ambient);
        this.layers.ambient.start();
        
        // Start noise
        if (preset.noise === 'white') {
            this.layers.noise = this.createWhiteNoise();
        } else if (preset.noise === 'pink') {
            this.layers.noise = this.createPinkNoise();
        } else {
            this.layers.noise = this.createBrownNoise();
        }
        
        const noiseGain = this.audioContext.createGain();
        noiseGain.gain.value = 0.1;
        this.layers.noise.connect(noiseGain);
        noiseGain.connect(this.masterGain);
        this.layers.noise.start();
        
        this.isPlaying = true;
        console.log(`Audio started: ${mood} mood`);
    }
    
    stop() {
        if (this.layers.binaural) {
            this.layers.binaural.stop();
        }
        if (this.layers.ambient) {
            this.layers.ambient.stop();
        }
        if (this.layers.noise) {
            this.layers.noise.stop();
        }
        
        this.isPlaying = false;
        console.log('Audio stopped');
    }
    
    changeMood(newMood) {
        if (this.isPlaying) {
            this.stop();
            setTimeout(() => {
                this.start(newMood);
            }, 100);
        }
    }
    
    setVolume(volume) {
        if (this.masterGain) {
            this.masterGain.gain.value = Math.max(0, Math.min(1, volume));
        }
    }
    
    // ============================================
    // 3D SPATIAL AUDIO
    // ============================================
    
    create3DSound(position, frequency = 440, type = 'sine') {
        // Create positioned audio source
        const osc = this.audioContext.createOscillator();
        const panner = this.audioContext.createPanner();
        const gain = this.audioContext.createGain();
        
        osc.frequency.value = frequency;
        osc.type = type;
        
        panner.panningModel = 'HRTF';
        panner.distanceModel = 'inverse';
        panner.refDistance = 1;
        panner.maxDistance = 100;
        panner.rolloffFactor = 1;
        
        panner.setPosition(position.x, position.y, position.z);
        
        gain.gain.value = 0.3;
        
        osc.connect(panner);
        panner.connect(gain);
        gain.connect(this.masterGain);
        
        return {
            oscillator: osc,
            panner: panner,
            gain: gain,
            start: () => osc.start(),
            stop: () => osc.stop(),
            updatePosition: (x, y, z) => panner.setPosition(x, y, z)
        };
    }
    
    updateListenerPosition(x, y, z) {
        if (this.audioContext && this.audioContext.listener) {
            this.audioContext.listener.setPosition(x, y, z);
        }
    }
}

// Export for use in game
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CoverFaceAudio;
}
'''

def get_audio_system_js():
    """Return the audio system JavaScript code."""
    return AUDIO_SYSTEM_JS
