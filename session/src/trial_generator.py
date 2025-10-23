"""
Trial Sequence Generator
=========================

Generates trial sequences for the P300-based CIT experiment with proper
randomization and counterbalancing.

Key features:
- Even rotation of probe/irrelevant images across trials
- Balanced target/nontarget distribution for S2
- Block-based organization
- Pseudo-randomization to avoid excessive repetition
"""

import random
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class TrialGenerator:
    """
    Generate trial sequences for P300-CIT experiment.
    
    Parameters
    ----------
    probe_images : list of str
        List of probe image file paths (can include multiple views)
    irrelevant_images : list of str
        List of irrelevant image file paths (can include multiple views)
    probe_reps : int
        Number of repetitions for probe OBJECT (distributed across views)
    irrelevant_reps : int
        Number of repetitions for each irrelevant OBJECT (distributed across views)
    target_proportion : float
        Proportion of S2 trials that are targets (0.0-1.0)
    num_blocks : int
        Number of blocks to divide trials into
    s2_target : str
        Target digit string for S2
    s2_nontargets : list of str
        Nontarget digit strings for S2
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        probe_images: List[str],
        irrelevant_images: List[str],
        probe_reps: int,
        irrelevant_reps: int,
        target_proportion: float,
        num_blocks: int,
        s2_target: str,
        s2_nontargets: List[str],
        seed: Optional[int] = None
    ):
        self.probe_images = probe_images
        self.irrelevant_images = irrelevant_images
        self.probe_reps = probe_reps
        self.irrelevant_reps = irrelevant_reps
        self.target_proportion = target_proportion
        self.num_blocks = num_blocks
        self.s2_target = s2_target
        self.s2_nontargets = s2_nontargets
        self.seed = seed
        
        self.logger = logging.getLogger(__name__)
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
        
        # Group images by object
        self.probe_objects = self._group_images_by_object(probe_images)
        self.irrelevant_objects = self._group_images_by_object(irrelevant_images)
        
        # Validate inputs
        self._validate_inputs()
    
    def _group_images_by_object(self, image_paths: List[str]) -> Dict[str, List[str]]:
        """
        Group image paths by object name.
        
        Images with same object but different views are grouped together.
        E.g., probe_pendrive_view1.jpg and probe_pendrive_view2.jpg -> {"pendrive": [...]}
        
        Parameters
        ----------
        image_paths : list of str
            List of image file paths
            
        Returns
        -------
        dict
            Dictionary mapping object names to lists of image paths
        """
        objects = {}
        
        for img_path in image_paths:
            obj_name = self._extract_object_name(img_path)
            
            if obj_name not in objects:
                objects[obj_name] = []
            
            objects[obj_name].append(img_path)
        
        return objects
    
    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if not self.probe_images:
            raise ValueError("No probe images provided")
        
        if not self.irrelevant_images:
            raise ValueError("No irrelevant images provided")
        
        if self.probe_reps <= 0:
            raise ValueError("probe_reps must be positive")
        
        if self.irrelevant_reps <= 0:
            raise ValueError("irrelevant_reps must be positive")
        
        if not (0 < self.target_proportion < 1):
            raise ValueError("target_proportion must be between 0 and 1")
        
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        
        # Calculate total trials (based on OBJECTS, not individual images)
        num_irrelevant_objects = len(self.irrelevant_objects)
        total_trials = self.probe_reps + num_irrelevant_objects * self.irrelevant_reps
        
        self.logger.info(
            f"Trial calculation: {self.probe_reps} probe + "
            f"{num_irrelevant_objects} objects × {self.irrelevant_reps} reps = "
            f"{total_trials} total trials"
        )
        
        if total_trials % self.num_blocks != 0:
            self.logger.warning(
                f"Total trials ({total_trials}) not evenly divisible by "
                f"num_blocks ({self.num_blocks}). Some blocks may have "
                "different numbers of trials."
            )
    
    def generate_trials(self) -> List[Dict]:
        """
        Generate complete trial list with S1 and S2 stimuli.
        
        Returns
        -------
        list of dict
            Trial list with keys: 'trial_num', 'block', 's1_image', 's1_type',
            's1_object', 's2_string', 's2_type'
        """
        self.logger.info("Generating trial sequence...")
        
        # Generate S1 stimuli with even rotation
        s1_trials = self._generate_s1_sequence()
        
        # Shuffle trials (with constraints)
        s1_trials = self._shuffle_with_constraints(s1_trials)
        
        # Assign blocks
        trials_per_block = len(s1_trials) // self.num_blocks
        
        for i, trial in enumerate(s1_trials):
            trial['block'] = (i // trials_per_block) + 1
            # Ensure last trials go to last block
            if trial['block'] > self.num_blocks:
                trial['block'] = self.num_blocks
        
        # Generate S2 stimuli
        s2_sequence = self._generate_s2_sequence(len(s1_trials))
        
        # Combine S1 and S2
        for i, (trial, s2) in enumerate(zip(s1_trials, s2_sequence)):
            trial['trial_num'] = i + 1
            trial['s2_string'] = s2['string']
            trial['s2_type'] = s2['type']
        
        self.logger.info(
            f"Generated {len(s1_trials)} trials across {self.num_blocks} blocks"
        )
        self._log_trial_statistics(s1_trials)
        
        return s1_trials
    
    def _generate_s1_sequence(self) -> List[Dict]:
        """
        Generate S1 stimulus sequence with even image/view rotation.
        
        For objects with multiple views, views are evenly rotated so that
        no single view appears more frequently than others.
        
        Returns
        -------
        list of dict
            S1 trial information
        """
        trials = []
        
        # Process probe object(s)
        # Note: typically there's only 1 probe object, but code handles multiple
        probe_trial_lists = []
        
        for obj_name, img_paths in self.probe_objects.items():
            obj_trials = []
            num_views = len(img_paths)
            
            # Distribute repetitions evenly across views
            reps_per_view = self.probe_reps // num_views
            remainder = self.probe_reps % num_views
            
            for i, img_path in enumerate(img_paths):
                # Give extra rep to first 'remainder' views to distribute evenly
                n_reps = reps_per_view + (1 if i < remainder else 0)
                
                for _ in range(n_reps):
                    obj_trials.append({
                        's1_image': img_path,
                        's1_type': 'probe',
                        's1_object': obj_name
                    })
            
            # Shuffle this object's trials
            random.shuffle(obj_trials)
            probe_trial_lists.append(obj_trials)
        
        # Process irrelevant objects
        irrelevant_trial_lists = []
        
        for obj_name, img_paths in self.irrelevant_objects.items():
            obj_trials = []
            num_views = len(img_paths)
            
            # Distribute repetitions evenly across views
            reps_per_view = self.irrelevant_reps // num_views
            remainder = self.irrelevant_reps % num_views
            
            for i, img_path in enumerate(img_paths):
                # Give extra rep to first 'remainder' views
                n_reps = reps_per_view + (1 if i < remainder else 0)
                
                for _ in range(n_reps):
                    obj_trials.append({
                        's1_image': img_path,
                        's1_type': 'irrelevant',
                        's1_object': obj_name
                    })
            
            # Shuffle this object's trials
            random.shuffle(obj_trials)
            irrelevant_trial_lists.append(obj_trials)
        
        # Interleave all object lists for even rotation across trial sequence
        # This ensures probe and irrelevants are distributed evenly
        all_lists = probe_trial_lists + irrelevant_trial_lists
        
        # Shuffle order of lists to avoid predictable patterns
        random.shuffle(all_lists)
        
        # Interleave trials from all lists
        while any(all_lists):
            for lst in all_lists:
                if lst:
                    trials.append(lst.pop(0))
        
        return trials
    
    def _shuffle_with_constraints(self, trials: List[Dict]) -> List[Dict]:
        """
        Shuffle trials with constraints to avoid excessive repetition.
        
        Parameters
        ----------
        trials : list of dict
            Trial list
            
        Returns
        -------
        list of dict
            Shuffled trial list
        """
        # Apply partial shuffling to avoid long runs of same stimulus
        # Use sliding window approach
        max_consecutive = 3  # Maximum consecutive trials with same object
        
        result = trials.copy()
        random.shuffle(result)
        
        # Check and fix consecutive repetitions
        improved = True
        iterations = 0
        max_iterations = 1000
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for i in range(len(result) - max_consecutive):
                # Check if we have max_consecutive same objects
                objects = [result[j]['s1_object'] for j in range(i, i + max_consecutive + 1)]
                
                if len(set(objects)) == 1:  # All same object
                    # Find a different trial to swap with
                    for j in range(i + max_consecutive + 1, len(result)):
                        if result[j]['s1_object'] != objects[0]:
                            # Swap
                            result[i + max_consecutive], result[j] = \
                                result[j], result[i + max_consecutive]
                            improved = True
                            break
                    
                    if improved:
                        break
        
        return result
    
    def _generate_s2_sequence(self, n_trials: int) -> List[Dict]:
        """
        Generate S2 stimulus sequence.
        
        Parameters
        ----------
        n_trials : int
            Number of trials
            
        Returns
        -------
        list of dict
            S2 stimulus information
        """
        # Calculate number of targets
        n_targets = int(n_trials * self.target_proportion)
        n_nontargets = n_trials - n_targets
        
        # Create S2 list
        s2_list = []
        
        # Add targets
        for _ in range(n_targets):
            s2_list.append({
                'string': self.s2_target,
                'type': 'target'
            })
        
        # Add nontargets (evenly distributed across nontarget strings)
        for i in range(n_nontargets):
            nontarget_idx = i % len(self.s2_nontargets)
            s2_list.append({
                'string': self.s2_nontargets[nontarget_idx],
                'type': 'nontarget'
            })
        
        # Shuffle with constraint: avoid consecutive targets
        random.shuffle(s2_list)
        
        # Fix consecutive targets
        for i in range(len(s2_list) - 1):
            if s2_list[i]['type'] == 'target' and s2_list[i + 1]['type'] == 'target':
                # Find next nontarget and swap
                for j in range(i + 2, len(s2_list)):
                    if s2_list[j]['type'] == 'nontarget':
                        s2_list[i + 1], s2_list[j] = s2_list[j], s2_list[i + 1]
                        break
        
        return s2_list
    
    def _extract_object_name(self, filepath: str) -> str:
        """
        Extract object name from image filename.
        
        Parameters
        ----------
        filepath : str
            Path to image file
            
        Returns
        -------
        str
            Object name (e.g., 'pendrive', 'mouse')
        """
        filename = Path(filepath).stem  # Get filename without extension
        
        # Expected format: probe_objectname_viewX or irr_objectname_viewX
        parts = filename.split('_')
        
        if len(parts) >= 3:
            # Join middle parts (in case object name has underscores)
            return '_'.join(parts[1:-1])
        elif len(parts) == 2:
            return parts[1]
        else:
            return filename
    
    def _log_trial_statistics(self, trials: List[Dict]) -> None:
        """Log statistics about generated trials."""
        # Count S1 types by object
        s1_object_counts = {}
        for trial in trials:
            obj = trial['s1_object']
            s1_type = trial['s1_type']
            key = f"{s1_type}_{obj}"
            s1_object_counts[key] = s1_object_counts.get(key, 0) + 1
        
        # Count S1 types by individual image (view)
        s1_image_counts = {}
        for trial in trials:
            img = Path(trial['s1_image']).name
            s1_image_counts[img] = s1_image_counts.get(img, 0) + 1
        
        # Count S2 types
        s2_target_count = sum(1 for t in trials if t['s2_type'] == 'target')
        s2_nontarget_count = sum(1 for t in trials if t['s2_type'] == 'nontarget')
        
        # Log
        self.logger.info("Trial statistics:")
        self.logger.info(f"  Total trials: {len(trials)}")
        self.logger.info(f"  Blocks: {self.num_blocks}")
        
        self.logger.info("  S1 distribution by object:")
        for key, count in sorted(s1_object_counts.items()):
            self.logger.info(f"    {key}: {count}")
        
        self.logger.info("  S1 distribution by image (view):")
        for img, count in sorted(s1_image_counts.items()):
            self.logger.info(f"    {img}: {count}")
        
        self.logger.info("  S2 distribution:")
        self.logger.info(f"    target: {s2_target_count} ({s2_target_count/len(trials)*100:.1f}%)")
        self.logger.info(f"    nontarget: {s2_nontarget_count} ({s2_nontarget_count/len(trials)*100:.1f}%)")

