# fhb_image_analysis_pipeline

Julian Cooper
Hirsch Lab
FHB Image Analysis Pipeline

Deep learning based RGB image analysis pipeline for quantifying Fusarium head blight on wheat. 

1. FHB_phenotyping_pipeline: Get disease inferences from images of wheat
- models: Deep learning models for image analysis pipeline
	- grain_head_and_other_detection: wheat spike detection
	- umn_full_growth_cycle_unet_128x128: wheat spike gradability
	- wheat_head_fhb_seg: wheat spike segmentation
	- wheat_spike_fhb_disease_non_gradable_classification_input_is_BGR: disease segmentation
- pipeline: Image analysis pipeline for analyzing images 

2. image_analysis_results: Compare pipeline disease inferences to field and manual image annotation 
- pipeline_field_plot: Compare pipeline results to average of five raters scoring disease in the field
- pipeline_n10K_plot: Compare pipeline results to five raters manually annotating disease on separate images at plot scale
- pipeline_n10K_spike: Compare pipeline results to five raters manually annotating disease on separate images at spike scale
- pipeline_n200_spike: Compare pipeline results to five raters manually annotating disease on same images at spike scale
- data_frames: Data for comparisons between different disease scoring methods
