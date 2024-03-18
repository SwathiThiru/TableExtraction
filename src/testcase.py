from inference import TableExtractionPipeline

# Create inference pipeline
pipe = TableExtractionPipeline(det_config_path='detection_config.json', det_model_path='../pubtables1m_detection_detr_r18.pth', det_device='cuda', str_config_path='structure_config.json', str_model_path='../pubtables1m_structure_detr_r18.pth', str_device='cuda')

# Recognize table(s) from image
extracted_tables = pipe.recognize(img, tokens, out_objects=True, out_cells=True, out_html=True, out_csv=True)

# Select table (there could be more than one)
extracted_table = extracted_tables[0]

# Get output in desired format
objects = extracted_table['objects']
cells = extracted_table['cells']
csv = extracted_table['csv']
html = extracted_table['html']