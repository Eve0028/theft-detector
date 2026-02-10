"""
P300-Based Concealed Information Test - Main Experiment
========================================================

This script runs the complete P300-based CIT experiment using the mock crime
paradigm. Participants view images (probe = stolen item, irrelevants = neutral
items) followed by digit strings while EEG is recorded.

Usage
-----
Run from command line::

    python experiment.py

Or with custom config::

    python experiment.py --config path/to/config.yaml

"""

import os
import sys
import argparse
import time
import csv
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import numpy as np
from psychopy import visual, core, event

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from trial_generator import TrialGenerator
from lsl_markers import LSLMarkerSender
from brainaccess_handler import BrainAccessHandler
from utils import (
    setup_logging,
    get_output_filename,
    find_image_files,
    get_timestamp
)


class P300_CIT_Experiment:
    """
    Main experiment class for P300-based Concealed Information Test.
    
    Parameters
    ----------
    config_path : str
        Path to experiment configuration YAML file
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.window: Optional[visual.Window] = None
        self.lsl_sender: Optional[LSLMarkerSender] = None
        self.eeg_handler: Optional[BrainAccessHandler] = None
        self.logger = None
        self.trial_list: List[Dict] = []
        self.data_file = None
        self.data_writer = None
        
        # Stimuli
        self.fixation = None
        self.image_stim = None
        self.text_stim = None
        
        # Timing
        self.clock = core.Clock()
        
        # Data collection
        self.behavioral_data = []
    
    def _load_config(self) -> Dict:
        """Load experiment configuration from YAML file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup(self) -> None:
        """Initialize experiment components."""
        # Setup logging
        log_dir = os.path.join(
            self.config['output']['data_directory'],
            self.config['output']['logs_subdirectory']
        )
        self.logger = setup_logging(
            log_dir,
            self.config['participant']['id'],
            self.config['participant']['session']
        )
        
        self.logger.info("="*60)
        self.logger.info("P300-BASED CONCEALED INFORMATION TEST")
        self.logger.info("="*60)
        self.logger.info(f"Participant: {self.config['participant']['id']}")
        self.logger.info(f"Session: {self.config['participant']['session']}")
        self.logger.info(f"Condition: {self.config['participant']['condition']}")
        self.logger.info(f"EEG Device: {self.config['eeg']['device_type']}")
        
        # Initialize LSL marker sender
        if self.config['eeg']['send_markers']:
            self.lsl_sender = LSLMarkerSender(
                stream_name=self.config['eeg']['lsl_stream_name'],
                stream_type=self.config['eeg']['lsl_stream_type'],
                stream_id=self.config['eeg']['lsl_stream_id'],
                device_type=self.config['eeg']['device_type'],
                enabled=True
            )
        else:
            self.lsl_sender = LSLMarkerSender(enabled=False)
        
        # Initialize BrainAccess EEG handler
        if self.config['eeg']['enabled'] and self.config['eeg']['device_type'] == 'brainaccess':
            ba_config = self.config['eeg']['brainaccess']
            self.eeg_handler = BrainAccessHandler(
                channels=ba_config['channels'],
                channel_mapping=ba_config.get('channel_mapping'),
                sampling_rate=ba_config['sampling_rate'],
                buffer_size=ba_config['buffer_size'],
                enabled=True,
                verbose=False  # Production mode - minimal logging overhead
            )
            
            # Try to connect to device
            self.logger.info("Connecting to BrainAccess device...")
            if self.eeg_handler.connect():
                self.logger.info("BrainAccess device ready")
                
                # Check signal quality (wait for buffer to fill)
                self.logger.info("Collecting data for signal quality check...")
                time.sleep(3)  # Wait for data (need 250 samples = 1s @ 250Hz)
                
                # Check buffer size
                buffer_samples = len(self.eeg_handler.eeg_data)
                self.logger.info(f"Buffer has {buffer_samples} samples")
                
                quality = self.eeg_handler.get_signal_quality()
                self.logger.info(f"Signal quality: {quality}")
                
                for ch, q in quality.items():
                    if q == 'poor':
                        self.logger.warning(f"Poor signal quality on channel {ch}")
            else:
                self.logger.error(
                    "Failed to connect to BrainAccess device. "
                    "Make sure BrainAccess Board is running."
                )
                # Continue anyway - user can retry
        else:
            self.eeg_handler = BrainAccessHandler(enabled=False)
        
        # Create PsychoPy window
        self.logger.info("Creating display window...")
        self.window = visual.Window(
            size=[1920, 1080],  # Will use full screen if fullscreen=True
            fullscr=self.config['display']['fullscreen'],
            screen=self.config['display']['screen_number'],
            color=self._normalize_color(self.config['display']['background_color']),
            units=self.config['display']['units'],
            allowGUI=False
        )
        
        # Create stimuli
        self._create_stimuli()
        
        # Generate trials
        self._generate_trials()
        
        # Setup data output
        self._setup_data_output()
        
        self.logger.info("Setup complete!")
    
    def _normalize_color(self, rgb: List[int]) -> List[float]:
        """Convert RGB [0, 255] to PsychoPy range [-1, 1]."""
        return [(c / 127.5) - 1 for c in rgb]
    
    def _create_stimuli(self) -> None:
        """Create visual stimuli."""
        # Fixation cross
        self.fixation = visual.TextStim(
            self.window,
            text='+',
            color=self._normalize_color(self.config['display']['fixation_color']),
            height=50
        )
        
        # Image stimulus (placeholder - will be set per trial)
        self.image_stim = visual.ImageStim(
            self.window,
            size=None  # Will be set based on config
        )
        
        # Text stimulus for S2 (digit strings)
        self.text_stim = visual.TextStim(
            self.window,
            text='',
            color=self._normalize_color(self.config['display']['text_color']),
            height=self.config['stimuli']['digits']['font_size']
        )
        
        # Instruction text
        self.instruction_text = visual.TextStim(
            self.window,
            text='',
            color=self._normalize_color(self.config['display']['text_color']),
            height=30,
            wrapWidth=1600
        )
    
    def _generate_trials(self) -> None:
        """Generate trial sequence."""
        self.logger.info("Generating trial sequence...")
        
        # Find image files
        image_dir = os.path.join(
            Path(self.config_path).parent.parent,
            'images'
        )
        
        probe_images, irrelevant_images = find_image_files(
            image_dir,
            use_normalized=self.config['stimuli']['images']['use_normalized']
        )
        
        self.logger.info(f"Found {len(probe_images)} probe images")
        self.logger.info(f"Found {len(irrelevant_images)} irrelevant images")
        
        # Create trial generator
        generator = TrialGenerator(
            probe_images=probe_images,
            irrelevant_images=irrelevant_images,
            probe_reps=self.config['trials']['probe_repetitions'],
            irrelevant_reps=self.config['trials']['irrelevant_repetitions'],
            target_proportion=self.config['trials']['target_proportion'],
            num_blocks=self.config['trials']['num_blocks'],
            s2_target=self.config['stimuli']['digits']['target'],
            s2_nontargets=self.config['stimuli']['digits']['nontargets']
        )
        
        # Generate trials
        self.trial_list = generator.generate_trials()
    
    def _setup_data_output(self) -> None:
        """Setup CSV file for behavioral data output."""
        output_dir = os.path.join(
            self.config['output']['data_directory'],
            self.config['output']['behavioral_subdirectory']
        )
        
        filename = get_output_filename(
            output_dir,
            self.config['participant']['id'],
            self.config['participant']['session'],
            suffix='behavioral'
        )
        
        self.logger.info(f"Output file: {filename}")
        
        # Open CSV file
        self.data_file = open(filename, 'w', newline='', encoding='utf-8')
        
        # Create CSV writer with all columns
        fieldnames = [
            'participant_id', 'session_id', 'condition', 'block', 'trial_index',
            'S1_type', 'S1_object', 'S1_filename',
            'fixation_onset_time', 'S1_onset_time', 'S1_response_key', 'S1_RT',
            'ISI_duration',
            'S2_type', 'S2_string', 'S2_onset_time', 'S2_response_key', 'S2_RT',
            'S2_correct',
            'ITI_duration',
            'LSL_fixation_marker', 'LSL_S1_marker', 'LSL_S1_response_marker',
            'LSL_S2_marker', 'LSL_S2_response_marker', 'LSL_ITI_marker',
            'timestamp_unix', 'timestamp_iso',
            'notes'
        ]
        
        self.data_writer = csv.DictWriter(self.data_file, fieldnames=fieldnames)
        self.data_writer.writeheader()
        self.data_file.flush()
        
        # Marker counter for behavioral CSV
        self.marker_counter = 0
    
    def _send_marker(self, marker: str) -> int:
        """
        Send marker using optimal method:
        1. Native BrainAccess SDK annotation (fast, <0.1ms)
        2. Optional LSL for Board recording (if enabled)
        
        Parameters
        ----------
        marker : str
            Marker string (e.g., "S1_onset_probe|trial=1")
            
        Returns
        -------
        int
            Marker sequence number for behavioral CSV
        """
        self.marker_counter += 1
        
        # Send to BrainAccess SDK (native, always if connected)
        if self.eeg_handler and self.eeg_handler.is_connected:
            self.eeg_handler.annotate(marker)
        
        # Send to LSL (optional, for Board app recording)
        if self.lsl_sender and self.lsl_sender.enabled:
            self.lsl_sender.send_marker(marker)
        
        return self.marker_counter
    
    def run(self) -> None:
        """Run the complete experiment."""
        try:
            # Show welcome instructions
            self._show_instructions('welcome')
            
            # Show main task instructions
            self._show_instructions('main_task')
            
            # Start EEG recording
            if self.eeg_handler and self.eeg_handler.is_connected:
                eeg_output_dir = os.path.join(
                    self.config['output']['data_directory'],
                    'eeg'
                )
                eeg_filename = get_output_filename(
                    eeg_output_dir,
                    self.config['participant']['id'],
                    self.config['participant']['session'],
                    suffix='raw',  # MNE convention: *_raw.fif
                    extension='fif'  # FIF format with embedded annotations
                )
                
                if self.eeg_handler.start_recording(eeg_filename):
                    self.logger.info(f"EEG recording started: {eeg_filename}")
                else:
                    self.logger.error("Failed to start EEG recording")
            
            # Run all blocks
            for block in range(1, self.config['trials']['num_blocks'] + 1):
                self._run_block(block)
                
                # Break between blocks (except after last block)
                if block < self.config['trials']['num_blocks']:
                    self._show_block_break(block)
            
            # Stop EEG recording
            if self.eeg_handler and self.eeg_handler.is_recording:
                self.eeg_handler.stop_recording()
                self.logger.info("EEG recording stopped")
            
            # Show end message
            self._show_instructions('end')
            
            self.logger.info("Experiment completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Experiment error: {e}", exc_info=True)
            raise
        
        finally:
            self.cleanup()
    
    def _run_block(self, block_num: int) -> None:
        """
        Run a single block of trials.
        
        Parameters
        ----------
        block_num : int
            Block number (1-indexed)
        """
        self.logger.info(f"Starting block {block_num}")
        
        # Send block start marker
        self._send_marker(f"block_start|block={block_num}")
        
        # Get trials for this block
        block_trials = [t for t in self.trial_list if t['block'] == block_num]
        
        # Run each trial
        for trial in block_trials:
            self._run_trial(trial)
            
            # Check for quit key
            if event.getKeys([self.config['keys']['quit_key']]):
                self.logger.warning("Experiment aborted by user")
                raise KeyboardInterrupt("User quit")
        
        # Send block end marker
        self._send_marker(f"block_end|block={block_num}")
        
        self.logger.info(f"Block {block_num} complete")
    
    def _run_trial(self, trial: Dict) -> None:
        """
        Run a single trial.
        
        Parameters
        ----------
        trial : dict
            Trial information from trial generator
        """
        trial_num = trial['trial_num']
        
        # Initialize trial data
        trial_data = {
            'participant_id': self.config['participant']['id'],
            'session_id': self.config['participant']['session'],
            'condition': self.config['participant']['condition'],
            'block': trial['block'],
            'trial_index': trial_num,
            'S1_type': trial['s1_type'],
            'S1_object': trial['s1_object'],
            'S1_filename': os.path.basename(trial['s1_image']),
            'S2_type': trial['s2_type'],
            'S2_string': trial['s2_string'],
            'notes': ''
        }
        
        # Get trial start timestamp
        trial_start_time, trial_start_iso = get_timestamp()
        trial_data['timestamp_unix'] = trial_start_time
        trial_data['timestamp_iso'] = trial_start_iso
        
        # Clear event buffer
        event.clearEvents()
        
        # 1. FIXATION CROSS
        self.clock.reset()
        fixation_onset = self.clock.getTime()
        trial_data['fixation_onset_time'] = fixation_onset
        
        # Send annotation (native SDK + optional LSL)
        marker_id = self._send_marker(
            f"fixation_onset|trial={trial_num}"
        )
        trial_data['LSL_fixation_marker'] = marker_id
        
        self.fixation.draw()
        self.window.flip()
        
        core.wait(self.config['timing']['fixation_duration'])
        
        # 2. S1 STIMULUS (IMAGE)
        self.clock.reset()
        s1_onset = self.clock.getTime()
        trial_data['S1_onset_time'] = s1_onset
        
        # Set image
        self.image_stim.setImage(trial['s1_image'])
        
        # Resize image to configured height while maintaining aspect ratio
        img_height = self.config['stimuli']['images']['image_height']
        self.image_stim.size = None  # Use image native size
        self.image_stim.size = img_height  # Set height, width auto-calculated
        
        marker_id = self._send_marker(
            f"S1_onset_{trial['s1_type']}|trial={trial_num},stim_id={trial['s1_object']}"
        )
        trial_data['LSL_S1_marker'] = marker_id
        
        self.image_stim.draw()
        self.window.flip()
        
        # Collect S1 response during image + ISI
        s1_duration = self.config['timing']['s1_duration']
        isi_duration = np.random.uniform(
            self.config['timing']['isi_min'],
            self.config['timing']['isi_max']
        )
        trial_data['ISI_duration'] = isi_duration
        
        # Wait for S1 duration
        core.wait(s1_duration)
        
        # 3. ISI (collect S1 response here)
        self.window.flip()  # Blank screen
        
        s1_response = None
        s1_rt = None
        
        # Wait during ISI and collect response
        isi_clock = core.Clock()
        while isi_clock.getTime() < isi_duration:
            keys = event.getKeys(
                keyList=[self.config['keys']['s1_response']],
                timeStamped=self.clock
            )
            if keys and s1_response is None:
                s1_response = keys[0][0]
                s1_rt = keys[0][1]
                
                # Send S1 response marker
                marker_id = self._send_marker(
                    f"S1_response|trial={trial_num},key={s1_response},rt={s1_rt:.4f}"
                )
                trial_data['LSL_S1_response_marker'] = marker_id
            
            core.wait(0.001)  # Small delay to prevent CPU overload
        
        trial_data['S1_response_key'] = s1_response if s1_response else 'None'
        trial_data['S1_RT'] = s1_rt if s1_rt is not None else np.nan
        
        # 4. S2 STIMULUS (DIGIT STRING)
        self.clock.reset()
        s2_onset = self.clock.getTime()
        trial_data['S2_onset_time'] = s2_onset
        
        self.text_stim.setText(trial['s2_string'])
        
        marker_id = self._send_marker(
            f"S2_onset_{trial['s2_type']}|trial={trial_num}"
        )
        trial_data['LSL_S2_marker'] = marker_id
        
        self.text_stim.draw()
        self.window.flip()
        
        # Wait for S2 duration
        s2_duration = self.config['timing']['s2_duration']
        core.wait(s2_duration)
        
        # Clear screen and collect S2 response
        self.window.flip()
        
        s2_response = None
        s2_rt = None
        
        response_window = self.config['timing']['s2_response_window']
        response_clock = core.Clock()
        
        while response_clock.getTime() < response_window:
            keys = event.getKeys(
                keyList=[
                    self.config['keys']['s2_target'],
                    self.config['keys']['s2_nontarget']
                ],
                timeStamped=self.clock
            )
            if keys and s2_response is None:
                s2_response = keys[0][0]
                s2_rt = keys[0][1]
                break
            
            core.wait(0.001)
        
        # Validate S2 response
        s2_correct = False
        if s2_response:
            if trial['s2_type'] == 'target':
                s2_correct = (s2_response == self.config['keys']['s2_target'])
            else:
                s2_correct = (s2_response == self.config['keys']['s2_nontarget'])
            
            # Send S2 response marker
            marker_id = self._send_marker(
                f"S2_response|trial={trial_num},key={s2_response},rt={s2_rt:.4f},correct={int(s2_correct)}"
            )
            trial_data['LSL_S2_response_marker'] = marker_id
        
        trial_data['S2_response_key'] = s2_response if s2_response else 'None'
        trial_data['S2_RT'] = s2_rt if s2_rt is not None else np.nan
        trial_data['S2_correct'] = int(s2_correct)
        
        # 5. ITI
        iti_duration = np.random.uniform(
            self.config['timing']['iti_min'],
            self.config['timing']['iti_max']
        )
        trial_data['ITI_duration'] = iti_duration
        
        marker_id = self._send_marker(f"ITI_start|trial={trial_num}")
        trial_data['LSL_ITI_marker'] = marker_id
        
        core.wait(iti_duration)
        
        # Save trial data
        self.data_writer.writerow(trial_data)
        self.data_file.flush()
    
    def _show_instructions(self, instruction_type: str) -> None:
        """
        Show instruction screen.
        
        Parameters
        ----------
        instruction_type : str
            Type of instruction: 'welcome', 'main_task', 'end'
        """
        text = self.config['instructions'][instruction_type]
        
        self.instruction_text.setText(text)
        self.instruction_text.draw()
        self.window.flip()
        
        # Wait for continue key
        event.waitKeys(keyList=[self.config['keys']['continue_key']])
    
    def _show_block_break(self, block_num: int) -> None:
        """
        Show break screen between blocks.
        
        Parameters
        ----------
        block_num : int
            Completed block number
        """
        break_duration = self.config['timing']['block_break_duration']
        total_blocks = self.config['trials']['num_blocks']
        
        # Format instruction text
        text = self.config['instructions']['block_break'].format(
            block_num=block_num,
            total_blocks=total_blocks,
            remaining_time=int(break_duration)
        )
        
        self.instruction_text.setText(text)
        self.instruction_text.draw()
        self.window.flip()
        
        # Wait for continue key or timeout
        break_clock = core.Clock()
        while break_clock.getTime() < break_duration:
            keys = event.getKeys(keyList=[self.config['keys']['continue_key']])
            if keys:
                break
            core.wait(0.1)
    
    def cleanup(self) -> None:
        """Clean up experiment resources."""
        self.logger.info("Cleaning up...")
        
        # Close data file
        if self.data_file:
            self.data_file.close()
        
        # Disconnect EEG device
        if self.eeg_handler:
            self.eeg_handler.disconnect()
        
        # Close LSL sender
        if self.lsl_sender:
            self.lsl_sender.close()
        
        # Close window
        if self.window:
            self.window.close()
        
        core.quit()


def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='P300-Based Concealed Information Test'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/experiment_config.yaml',
        help='Path to experiment configuration file'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Relative to project root (parent of src/)
        project_root = Path(__file__).parent.parent
        config_path = project_root / config_path
    
    # Check if config exists
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Create and run experiment
    experiment = P300_CIT_Experiment(str(config_path))
    experiment.setup()
    experiment.run()


if __name__ == '__main__':
    main()

