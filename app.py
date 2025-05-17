import gradio as gr
from PIL import Image, ImageDraw 
import os
import traceback 
import inspect 
import numpy as np 
import io 
import uuid 
import shutil 
import zipfile 

# --- Global Configurations ---
MODELS_CONFIG = {
    "Rembg - U2Net (General)": {"library": "rembg", "function_kwargs": {"model_name": "u2net"}},
    "Rembg - U2NetP (Lightweight)": {"library": "rembg", "function_kwargs": {"model_name": "u2netp"}},
    "Rembg - ISNet (General Use)": {"library": "rembg", "function_kwargs": {"model_name": "isnet-general-use"}},
    "Rembg - SAM (Segment Anything)": {"library": "rembg", "function_kwargs": {"model_name": "sam", "sam_model_type": "vit_b"}},
    "TransparentBG - Base (Old API)": {
        "library": "transparent_background", "init_kwargs": {"mode": "base", "device": "cpu", "jit": False}, "process_kwargs": {"type": "rgba"} 
    },
    "TransparentBG - Fast (Old API)": {
        "library": "transparent_background", "init_kwargs": {"mode": "fast", "device": "cpu", "jit": False}, "process_kwargs": {"type": "rgba"}
    }
}
first_model_run_messages = {model_name: True for model_name in MODELS_CONFIG}
SUPPORTED_DOWNLOAD_FORMATS = {
    "PNG (recommended for transparency)": "PNG", "JPEG (no transparency, smaller size)": "JPEG", "WEBP (good compression, supports transparency)": "WEBP"
}
DEFAULT_DOWNLOAD_FORMAT_KEY = "PNG (recommended for transparency)"
TEMP_OUTPUT_DIR = "temp_outputs_gradio" 

# --- Helper Function for AI Model Processing ---
def _process_single_image_with_model(input_pil_image_rgba, model_key_for_log, config_details):
    output_pil_image_single = None; print(f"--- Helper: AI Processing '{model_key_for_log}' ---")
    if not isinstance(input_pil_image_rgba, Image.Image): print(f"ERROR (Helper): Invalid input for {model_key_for_log}: {type(input_pil_image_rgba)}"); return None
    if input_pil_image_rgba.mode != "RGBA": 
        input_pil_image_rgba = input_pil_image_rgba.convert("RGBA")
    img_to_process_for_model = input_pil_image_rgba.copy()
    if config_details["library"] == "rembg":
        from rembg import remove as rembg_default_remove, new_session as rembg_new_session
        actual_rembg_model_identifier = config_details.get("function_kwargs", {}).get("model_name")
        if not actual_rembg_model_identifier: return None
        REMBG_API_DEFAULT_MODEL = "u2net"
        if actual_rembg_model_identifier == REMBG_API_DEFAULT_MODEL:
            additional_kwargs_for_default = {k:v for k,v in config_details.get("function_kwargs", {}).items() if k != "model_name"}
            output_pil_image_single = rembg_default_remove(img_to_process_for_model, **additional_kwargs_for_default) if additional_kwargs_for_default else rembg_default_remove(img_to_process_for_model)
        else: 
            session_kwargs = {k:v for k,v in config_details.get("function_kwargs", {}).items() if k != "model_name"}
            session = rembg_new_session(model_name=actual_rembg_model_identifier, **session_kwargs)
            prediction_results = session.predict(img_to_process_for_model)
            if prediction_results:
                raw_output_from_predict = prediction_results[0]
                if isinstance(raw_output_from_predict, Image.Image):
                    if raw_output_from_predict.mode == 'L': 
                        mask_to_apply = raw_output_from_predict
                        if img_to_process_for_model.size != mask_to_apply.size: mask_to_apply = mask_to_apply.resize(img_to_process_for_model.size, Image.Resampling.LANCZOS)
                        original_rgba_base = input_pil_image_rgba.copy() 
                        if original_rgba_base.size != mask_to_apply.size: original_rgba_base = original_rgba_base.resize(mask_to_apply.size, Image.Resampling.LANCZOS)
                        original_rgba_base.putalpha(mask_to_apply); output_pil_image_single = original_rgba_base
                    elif raw_output_from_predict.mode == 'RGBA': output_pil_image_single = raw_output_from_predict
                    else: 
                        try: output_pil_image_single = raw_output_from_predict.convert("RGBA")
                        except Exception as e_conv: print(f"Rembg conv err: {e_conv}")
    elif config_details["library"] == "transparent_background":
        from transparent_background import Remover as TB_Remover
        img_for_tb_rgb = img_to_process_for_model.convert("RGB") 
        init_kwargs_for_tb = config_details.get("init_kwargs", {}); remover = TB_Remover(**init_kwargs_for_tb) 
        process_kwargs_for_tb = config_details.get("process_kwargs", {}); 
        processed_output_tb = remover.process(img_for_tb_rgb, **process_kwargs_for_tb)
        if isinstance(processed_output_tb, Image.Image):
            if processed_output_tb.mode != "RGBA": output_pil_image_single = processed_output_tb.convert("RGBA")
            else: output_pil_image_single = processed_output_tb
        elif isinstance(processed_output_tb, np.ndarray):
            try:
                if processed_output_tb.ndim == 3 and processed_output_tb.shape[2] == 4: output_pil_image_single = Image.fromarray(processed_output_tb, 'RGBA')
                elif processed_output_tb.ndim == 3 and processed_output_tb.shape[2] == 3: output_pil_image_single = Image.fromarray(processed_output_tb, 'RGB').convert("RGBA")
                else: output_pil_image_single = Image.fromarray(processed_output_tb).convert("RGBA")
            except Exception as e_conv: print(f"TB NP conv err for '{model_key_for_log}': {e_conv}")
    if output_pil_image_single and output_pil_image_single.mode != "RGBA":
        output_pil_image_single = output_pil_image_single.convert("RGBA")
    print(f"--- Helper: Finished '{model_key_for_log}'. Output mode: {output_pil_image_single.mode if output_pil_image_single else 'None'} ---")
    return output_pil_image_single

def prepare_image_for_download(image_to_download_pil, download_format_key, base_filename="processed_image"):
    if not isinstance(image_to_download_pil, Image.Image):
        if image_to_download_pil is None: return None, "Error: No image available for download."
        print(f"WARN: prepare_image_for_download received non-PIL: {type(image_to_download_pil)}. This is unexpected.")
        return None, "Error: Invalid image data for download (not a PIL Image)."
    if image_to_download_pil.mode != "RGBA" and "A" not in image_to_download_pil.mode :
        image_to_download_pil = image_to_download_pil.convert("RGBA")
    os.makedirs(os.path.join(TEMP_OUTPUT_DIR, "downloads"), exist_ok=True)
    actual_download_format = SUPPORTED_DOWNLOAD_FORMATS.get(download_format_key, "PNG")
    file_extension = actual_download_format.lower(); temp_file_path = None
    if actual_download_format == "JPEG": file_extension = "jpg"
    unique_id = uuid.uuid4().hex[:8]
    temp_file_path = os.path.join(TEMP_OUTPUT_DIR, "downloads", f"{base_filename}_{unique_id}.{file_extension}")
    save_kwargs = {}; image_to_save = image_to_download_pil.copy()
    if actual_download_format == 'JPEG':
        if image_to_save.mode != 'RGB': 
            background = Image.new("RGB", image_to_save.size, (255, 255, 255))
            try: alpha = image_to_save.getchannel('A'); background.paste(image_to_save, mask=alpha)
            except: background.paste(image_to_save) 
            image_to_save = background
        save_kwargs['quality'] = 95 
    elif actual_download_format == 'WEBP':
        save_kwargs['quality'] = 90; save_kwargs['lossless'] = 'A' in image_to_save.mode
    try:
        image_to_save.save(temp_file_path, format=actual_download_format, **save_kwargs)
        return temp_file_path, f"Ready to download as {actual_download_format}."
    except Exception as e_save: traceback.print_exc(); return None, f"Error saving: {e_save}"

# --- Tab 1: Single Model Processing ---
def tab1_run_ai_model_func(initial_upload_pil_from_state, selected_model_key):
    print(f"\n--- tab1_run_ai_model_func: Selected {selected_model_key} ---")
    if not isinstance(initial_upload_pil_from_state, Image.Image): 
        return None, "Upload an image globally first (at the top).", None 
    global first_model_run_messages
    if first_model_run_messages.get(selected_model_key, True):
        print(f"Status for {selected_model_key}: Downloading/loading model..."); first_model_run_messages[selected_model_key] = False
    ai_processed_image = _process_single_image_with_model(initial_upload_pil_from_state, selected_model_key, MODELS_CONFIG[selected_model_key])
    tab1_ai_preview_state_output = ai_processed_image.copy() if ai_processed_image else None
    status_msg = f"AI ({selected_model_key}) failed."
    if ai_processed_image:
        status_msg = f"AI ({selected_model_key}) complete. Preview shown. Ready to Send to Inpaint or Download."
    print(f"DEBUG tab1_run_ai: AI status: {status_msg}. tab1_ai_preview_state_output type: {type(tab1_ai_preview_state_output)}")
    return ai_processed_image, status_msg, tab1_ai_preview_state_output

# --- Tab 2: All Models Comparison ---
def tab2_run_all_models_func(initial_upload_pil_from_state): 
    print(f"\n--- tab2_run_all_models_func ---")
    if not isinstance(initial_upload_pil_from_state, Image.Image): 
        return [], "Upload an image globally first (at the top)." 
    os.makedirs(os.path.join(TEMP_OUTPUT_DIR, "gallery"), exist_ok=True)
    gallery_items = []; status_list = []; global first_model_run_messages
    for model_key, config in MODELS_CONFIG.items():
        print(f"Gallery Processing: {model_key}")
        if first_model_run_messages.get(model_key, True): first_model_run_messages[model_key] = False
        img_out = _process_single_image_with_model(initial_upload_pil_from_state, model_key, config)
        if img_out:
            fname = f"{model_key.replace(' ','_').replace(':','').replace('(','').replace(')','').replace('/','_')}_{uuid.uuid4().hex[:6]}.png"
            fpath = os.path.join(TEMP_OUTPUT_DIR, "gallery", fname)
            img_out.save(fpath, "PNG"); gallery_items.append((fpath, model_key))
            status_list.append(f"{model_key}:OK")
        else: status_list.append(f"{model_key}:Fail")
    return gallery_items, "All models processed: " + " | ".join(status_list)

def tab2_download_all_gallery_func(gallery_list_data, download_format_key):
    print(f"\n--- tab2_download_all_gallery_func ---");
    if not gallery_list_data: return None, "No gallery images to download."
    image_paths_to_zip = []
    if gallery_list_data and isinstance(gallery_list_data[0], dict) and gallery_list_data[0].get('image') and isinstance(gallery_list_data[0]['image'], dict) and gallery_list_data[0]['image'].get('path'):
        image_paths_to_zip = [item['image']['path'] for item in gallery_list_data if item.get('image') and item['image'].get('path')]
    elif gallery_list_data and isinstance(gallery_list_data[0], tuple):
        image_paths_to_zip = [item[0] for item in gallery_list_data]
    else:
        print(f"WARN: Gallery data format not recognized for zipping. Data: {str(gallery_list_data)[:200]}")
        return None, "Gallery data format not recognized for zipping."
    if not image_paths_to_zip: return None, "No valid image paths found in gallery data."
    zip_filename_base = f"all_models_output_{uuid.uuid4().hex[:8]}.zip"; zip_filepath = os.path.join(TEMP_OUTPUT_DIR, "downloads", zip_filename_base)
    os.makedirs(os.path.join(TEMP_OUTPUT_DIR, "downloads"), exist_ok=True)
    with zipfile.ZipFile(zip_filepath, 'w') as zipf:
        for img_path in image_paths_to_zip:
            if os.path.exists(img_path):
                try:
                    img_to_convert = Image.open(img_path).convert("RGBA"); actual_dl_format = SUPPORTED_DOWNLOAD_FORMATS.get(download_format_key, "PNG")
                    file_ext = actual_dl_format.lower(); base_img_name = os.path.splitext(os.path.basename(img_path))[0]
                    if actual_dl_format == "JPEG": file_ext = "jpg"
                    converted_filename_in_zip = f"{base_img_name}_as_{actual_dl_format.lower()}.{file_ext}"
                    byte_io = io.BytesIO(); save_kwargs_zip = {}; img_to_save_in_zip = img_to_convert.copy()
                    if actual_dl_format == 'JPEG':
                        if img_to_save_in_zip.mode != 'RGB': background = Image.new("RGB", img_to_save_in_zip.size, (255,255,255)); alpha = img_to_save_in_zip.getchannel('A'); background.paste(img_to_save_in_zip, mask=alpha); img_to_save_in_zip = background
                        save_kwargs_zip['quality'] = 95
                    elif actual_dl_format == 'WEBP': save_kwargs_zip['quality'] = 90; save_kwargs_zip['lossless'] = 'A' in img_to_save_in_zip.mode
                    img_to_save_in_zip.save(byte_io, format=actual_dl_format, **save_kwargs_zip)
                    byte_io.seek(0); zipf.writestr(converted_filename_in_zip, byte_io.read())
                except Exception as e_zip_item: print(f"Error processing {img_path} for ZIP: {e_zip_item}")
    return zip_filepath, "ZIP file of gallery images (converted) ready."

# --- Tab 3: Inpaint/Editor Logic ---
def send_to_inpaint_editor_func(source_image_data_for_editor, initial_upload_pil_for_restore, status_prefix="Loaded"): # Renamed first param
    print(f"\n--- send_to_inpaint_editor_func ---")
    print(f"DEBUG In send_to_inpaint: source_image_data_for_editor type: {type(source_image_data_for_editor)}")
    print(f"DEBUG In send_to_inpaint: initial_upload_pil_for_restore type: {type(initial_upload_pil_for_restore)}")

    active_pil_image = None
    if isinstance(source_image_data_for_editor, Image.Image): 
        active_pil_image = source_image_data_for_editor.copy() 
    elif isinstance(source_image_data_for_editor, str) and os.path.exists(source_image_data_for_editor):
        try: active_pil_image = Image.open(source_image_data_for_editor)
        except Exception as e_load: return gr.Tabs(selected="tab_single_model"), None, f"Error loading image from path: {e_load}", None, None, None
    
    if not active_pil_image: return gr.Tabs(selected="tab_single_model"), None, "Error: No valid image data to send to editor.", None, None, None
    if active_pil_image.mode != "RGBA": active_pil_image = active_pil_image.convert("RGBA")
    
    editor_value_output = {"background": active_pil_image.copy(), "layers": [], "composite": active_pil_image.copy()}
    
    current_initial_upload_pil = None
    if isinstance(initial_upload_pil_for_restore, Image.Image):
        current_initial_upload_pil = initial_upload_pil_for_restore.copy()
        if current_initial_upload_pil.mode != "RGBA": current_initial_upload_pil = current_initial_upload_pil.convert("RGBA")
    elif initial_upload_pil_for_restore is not None: 
        print(f"WARN: initial_upload_pil_for_restore was not PIL in send_to_inpaint: {type(initial_upload_pil_for_restore)}.")
            
    active_editor_state_output = active_pil_image.copy() 
    initial_upload_state_output_for_tab3 = current_initial_upload_pil 
    original_preview_for_tab3 = current_initial_upload_pil.copy() if current_initial_upload_pil else None

    print(f"DEBUG Out send_to_inpaint: active_editor_state_output type: {type(active_editor_state_output)}")
    print(f"DEBUG Out send_to_inpaint: initial_upload_state_output_for_tab3 type: {type(initial_upload_state_output_for_tab3)}")
    return gr.Tabs(selected="tab_inpaint_editor"), editor_value_output, f"{status_prefix} to Inpaint Editor.", \
           active_editor_state_output, initial_upload_state_output_for_tab3, original_preview_for_tab3

def tab3_apply_manual_edits_func(active_editor_img_from_state_pil, initial_upload_from_state_pil, editor_data_dict, edit_mode_str):
    print(f"\n--- tab3_apply_manual_edits_func ---")
    print(f"DEBUG In apply_edits: active_editor_img_from_state_pil type: {type(active_editor_img_from_state_pil)}")
    print(f"DEBUG In apply_edits: initial_upload_from_state_pil type: {type(initial_upload_from_state_pil)}")
    if not isinstance(active_editor_img_from_state_pil, Image.Image):
        err_msg = f"Error: Active editor image invalid (type: {type(active_editor_img_from_state_pil)}). Load image."
        current_editor_content = editor_data_dict if isinstance(editor_data_dict, dict) else None
        return current_editor_content, err_msg, active_editor_img_from_state_pil, initial_upload_from_state_pil
    active_editor_img_pil = active_editor_img_from_state_pil.copy();
    if active_editor_img_pil.mode != "RGBA": active_editor_img_pil = active_editor_img_pil.convert("RGBA")
    if "Restore" in edit_mode_str and not isinstance(initial_upload_from_state_pil, Image.Image):
        err_msg = f"Error: Original for restore invalid (type: {type(initial_upload_from_state_pil)})."
        return {"background":active_editor_img_pil, "layers":[], "composite":active_editor_img_pil}, err_msg, active_editor_img_pil, initial_upload_from_state_pil
    if not editor_data_dict or not isinstance(editor_data_dict, dict) or not editor_data_dict.get("layers") or not editor_data_dict["layers"][0] or not isinstance(editor_data_dict["layers"][0], Image.Image):
        return {"background":active_editor_img_pil, "layers":[], "composite":active_editor_img_pil}, "No new strokes. Paint first.", active_editor_img_pil, initial_upload_from_state_pil
    base_for_this_edit_pil = active_editor_img_pil; stroke_layer_pil = editor_data_dict["layers"][0]
    if base_for_this_edit_pil.size != stroke_layer_pil.size: stroke_layer_pil = stroke_layer_pil.resize(base_for_this_edit_pil.size, Image.Resampling.NEAREST)
    if stroke_layer_pil.mode != 'RGBA': stroke_layer_pil = stroke_layer_pil.convert('RGBA')
    stroke_alpha_np = np.array(stroke_layer_pil.getchannel('A')); user_painted_mask_np = stroke_alpha_np > 10 
    current_image_np = np.array(base_for_this_edit_pil); new_rgb_np = current_image_np[:, :, :3].copy(); new_alpha_np = current_image_np[:, :, 3].copy()
    if "Erase" in edit_mode_str: new_alpha_np[user_painted_mask_np] = 0 
    elif "Restore" in edit_mode_str:
        original_pil_for_restore_pixels = initial_upload_from_state_pil.copy()
        if original_pil_for_restore_pixels.size != base_for_this_edit_pil.size: original_pil_for_restore_pixels = original_pil_for_restore_pixels.resize(base_for_this_edit_pil.size, Image.Resampling.LANCZOS)
        if original_pil_for_restore_pixels.mode != "RGBA": original_pil_for_restore_pixels = original_pil_for_restore_pixels.convert("RGBA")
        original_np_for_restore_pixels = np.array(original_pil_for_restore_pixels)
        new_alpha_np[user_painted_mask_np] = 255; new_rgb_np[user_painted_mask_np] = original_np_for_restore_pixels[user_painted_mask_np, :3]
    refined_pil_image = Image.fromarray(np.dstack((new_rgb_np, new_alpha_np)), 'RGBA')
    new_editor_value_output = {"background": refined_pil_image.copy(), "layers": [], "composite": refined_pil_image.copy()}
    active_editor_state_output = refined_pil_image.copy(); initial_upload_state_output = initial_upload_from_state_pil 
    print(f"DEBUG Out apply_edits: active_editor_state_output type: {type(active_editor_state_output)}")
    return new_editor_value_output, "Manual edits applied.", active_editor_state_output, initial_upload_state_output

# --- UI Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown("# 🎨 AI Background Remover & Editor 🖼️")
    gr.Markdown("Global image uploader at the top. Processed images can be sent to the Inpaint tab for manual refinement.")
    with gr.Row():
        with gr.Column(scale=1, min_width=350): 
            global_input_image_uploader = gr.Image(type="pil", label="UPLOAD IMAGE HERE", sources=["upload", "clipboard", "webcam"], height=300) 
        with gr.Column(scale=2):
            master_status_component = gr.Textbox(label="Overall Status", interactive=False, lines=3, max_lines=5, value="Welcome! Upload an image above to start.")
    gr.Markdown("---") 
    initial_upload_state = gr.State(value=None)      
    active_editor_image_state = gr.State(value=None) 
    tab1_ai_preview_intermediate_state = gr.State(value=None) 
    tab2_selected_gallery_image_path_state = gr.State(None) 
    with gr.Tabs() as main_tabs:
        with gr.TabItem("1. Process Single Image", id="tab_single_model"):
            gr.Markdown("## Step 1: Process with a Specific AI Model")
            gr.Markdown("Uses the globally uploaded image.")
            with gr.Row():
                with gr.Column(scale=1):
                    tab1_model_selector = gr.Dropdown(choices=list(MODELS_CONFIG.keys()), value=list(MODELS_CONFIG.keys())[0] if MODELS_CONFIG else None, label="Select AI Model")
                    tab1_run_ai_btn = gr.Button("Run Selected AI Model", variant="primary")
                with gr.Column(scale=1):
                    tab1_ai_preview_display = gr.Image(type="pil", label="AI Processed Preview", interactive=False) 
            with gr.Row():
                tab1_send_to_inpaint_btn = gr.Button("Send AI Preview to Inpaint/Editor Tab")
            with gr.Accordion("Download AI Preview", open=False):
                tab1_dl_format_selector = gr.Dropdown(choices=list(SUPPORTED_DOWNLOAD_FORMATS.keys()), value=DEFAULT_DOWNLOAD_FORMAT_KEY, label="Download Format")
                tab1_download_ai_preview_btn = gr.Button("Prepare Download")
                tab1_download_ai_preview_file = gr.File(label="Download Link")
        with gr.TabItem("2. All Models Comparison", id="tab_all_models"):
            gr.Markdown("## Step 2: Compare Results from All AI Models")
            gr.Markdown("Uses the globally uploaded image. Click 'Process' then select from gallery.")
            tab2_run_all_btn = gr.Button("Process with ALL Models", variant="primary") # Removed preview here
            tab2_gallery = gr.Gallery(label="All Model Results", show_label=False, columns=[3], object_fit="contain", height="auto", preview=True)
            gr.Markdown("Click an image in the gallery to select it.")
            with gr.Row():
                tab2_send_selected_to_inpaint_btn = gr.Button("Send Selected Gallery Image to Inpaint/Editor")
            with gr.Accordion("Download Gallery Images", open=False):
                tab2_dl_format_selector_gallery = gr.Dropdown(choices=list(SUPPORTED_DOWNLOAD_FORMATS.keys()), value=DEFAULT_DOWNLOAD_FORMAT_KEY, label="Format for Gallery Download(s)")
                tab2_download_selected_btn = gr.Button("Download Selected Gallery Image")
                tab2_download_all_zip_btn = gr.Button("Download All Gallery Images as ZIP")
                tab2_download_gallery_file = gr.File(label="Download Link")
        with gr.TabItem("3. Inpaint / Manual Editor", id="tab_inpaint_editor"):
            gr.Markdown("## Step 3: Fine-tune with Inpainting Tools")
            gr.Markdown("Image loaded from Tab 1 or Tab 2. Use editor tools (top-left of image canvas) for brush/eraser & size. Color choice for brush doesn't affect erase/restore logic.")
            with gr.Row():
                with gr.Column(scale=1): 
                    gr.Markdown("#### Original Image Reference")
                    tab3_original_preview_display = gr.Image(type="pil", label="Original (for restore reference)", interactive=False, height=600)
                with gr.Column(scale=2): 
                    gr.Markdown("#### Editor Canvas")
                    tab3_editor = gr.ImageEditor(
                        label="Paint on this image. Eraser icon for 'Erase', Brush icon for 'Restore'.",
                        type="pil", height=600, interactive=True,
                        brush=gr.Brush(default_size=20, colors=["#00FF00"], color_mode="fixed"), 
                        eraser=gr.Eraser(default_size=20) 
                    )
            with gr.Row(equal_height=True): 
                with gr.Column(scale=1):
                    tab3_edit_mode = gr.Radio(["Erase (make transparent)", "Restore (make opaque from original)"], label="Current Paint Mode Intention", value="Erase (make transparent)")
                with gr.Column(scale=1):
                    tab3_apply_edits_btn = gr.Button("Apply Painted Edits", variant="primary", scale=2) 
            with gr.Accordion("Download Edited Image", open=True):
                tab3_dl_format_selector = gr.Dropdown(choices=list(SUPPORTED_DOWNLOAD_FORMATS.keys()), value=DEFAULT_DOWNLOAD_FORMAT_KEY, label="Download Format")
                tab3_download_edited_btn = gr.Button("Prepare Download")
                tab3_download_edited_file = gr.File(label="Download Link")

    # --- Event Handlers ---
    def handle_global_upload_event(uploaded_image_pil):
        if uploaded_image_pil:
            pil_rgba_image = uploaded_image_pil.copy().convert("RGBA")
            # Outputs: initial_state, status, 
            #          (clear these) tab1_ai_preview_display, tab1_ai_preview_intermediate_state, 
            #          tab2_gallery, tab3_editor, active_editor_image_state, 
            #          all download files
            return pil_rgba_image, "New image uploaded. Select an action from the tabs below.", \
                   None, None, \
                   [], None, None, \
                   None, None, None 
        return None, "Upload cleared.", None, None, [], None, None, None, None, None
    global_input_image_uploader.upload(
        fn=handle_global_upload_event, inputs=[global_input_image_uploader],
        outputs=[initial_upload_state, master_status_component,
                 tab1_ai_preview_display, tab1_ai_preview_intermediate_state, 
                 tab2_gallery, tab3_editor, active_editor_image_state, 
                 tab1_download_ai_preview_file, tab2_download_gallery_file, tab3_download_edited_file]
    )
    
    tab1_run_ai_btn.click(
        fn=tab1_run_ai_model_func, inputs=[initial_upload_state, tab1_model_selector],
        outputs=[tab1_ai_preview_display, master_status_component, tab1_ai_preview_intermediate_state] 
    ).then(fn=lambda: (None, []), outputs=[tab1_download_ai_preview_file, tab2_gallery])

    tab1_send_to_inpaint_btn.click(
        fn=send_to_inpaint_editor_func, 
        inputs=[tab1_ai_preview_intermediate_state, initial_upload_state], 
        outputs=[main_tabs, tab3_editor, master_status_component, 
                 active_editor_image_state, initial_upload_state, 
                 tab3_original_preview_display] 
    ).then(fn=lambda: None, outputs=[tab1_download_ai_preview_file])

    tab1_download_ai_preview_btn.click(
        fn=prepare_image_for_download, inputs=[tab1_ai_preview_intermediate_state, tab1_dl_format_selector], 
        outputs=[tab1_download_ai_preview_file, master_status_component]
    )

    tab2_run_all_btn.click(
        fn=tab2_run_all_models_func, inputs=[initial_upload_state],
        outputs=[tab2_gallery, master_status_component] # Corrected: 2 outputs expected
    ).then(fn=lambda: (None, None), outputs=[tab2_selected_gallery_image_path_state, tab2_download_gallery_file])

    def handle_gallery_select_event_wrapper_tab2(evt: gr.SelectData):
        print(f"DEBUG Gallery Select Event: evt.selected={evt.selected}, evt.value type={type(evt.value)}, value={str(evt.value)[:200]}")
        selected_filepath = None
        if evt.selected:
            if isinstance(evt.value, dict): # Check if it's the dict structure
                image_data = evt.value.get('image', evt.value) # Try 'image' key, else evt.value itself
                if isinstance(image_data, dict) and 'path' in image_data:
                    selected_filepath = image_data['path']
                elif isinstance(image_data, str): # If image_data was directly the path
                     selected_filepath = image_data
            elif isinstance(evt.value, str): # If evt.value itself is the path string
                selected_filepath = evt.value
            
            if selected_filepath and os.path.exists(selected_filepath):
                print(f"DEBUG Gallery Select: Extracted filepath: {selected_filepath}")
                return selected_filepath, f"Selected '{os.path.basename(selected_filepath)}' from gallery."
            else:
                print(f"WARN: Gallery selection - could not extract valid filepath from evt.value: {str(evt.value)[:200]}")
                return None, "Could not identify file from gallery selection."
        return None, "Gallery selection cleared."
    tab2_gallery.select(handle_gallery_select_event_wrapper_tab2, outputs=[tab2_selected_gallery_image_path_state, master_status_component])

    tab2_send_selected_to_inpaint_btn.click(
        fn=send_to_inpaint_editor_func, 
        inputs=[tab2_selected_gallery_image_path_state, initial_upload_state], 
        outputs=[main_tabs, tab3_editor, master_status_component, 
                 active_editor_image_state, initial_upload_state, 
                 tab3_original_preview_display] 
    ).then(fn=lambda: None, outputs=[tab2_download_gallery_file])

    def download_selected_gallery_func(sel_gallery_path, dl_format):
        if not sel_gallery_path: return None, "Select an image from gallery."
        try: img = Image.open(sel_gallery_path).convert("RGBA")
        except: return None, "Error loading selected gallery image."
        return prepare_image_for_download(img, dl_format, base_filename="gallery_sel")
    tab2_download_selected_btn.click(download_selected_gallery_func, [tab2_selected_gallery_image_path_state, tab2_dl_format_selector_gallery], [tab2_download_gallery_file, master_status_component])

    tab2_download_all_zip_btn.click(
        fn=tab2_download_all_gallery_func, inputs=[tab2_gallery, tab2_dl_format_selector_gallery],
        outputs=[tab2_download_gallery_file, master_status_component]
    )
    
    tab3_apply_edits_btn.click(
        fn=tab3_apply_manual_edits_func,
        inputs=[active_editor_image_state, initial_upload_state, tab3_editor, tab3_edit_mode],
        outputs=[tab3_editor, master_status_component, active_editor_image_state, initial_upload_state]
    ).then(fn=lambda: None, outputs=[tab3_download_edited_file])

    tab3_download_edited_btn.click(
        fn=prepare_image_for_download, inputs=[active_editor_image_state, tab3_dl_format_selector],
        outputs=[tab3_download_edited_file, master_status_component]
    )

def cleanup_temp_files():
    if os.path.exists(TEMP_OUTPUT_DIR):
        try: shutil.rmtree(TEMP_OUTPUT_DIR)
        except Exception as e: print(f"Error cleaning {TEMP_OUTPUT_DIR}: {e}")
    try: 
        os.makedirs(os.path.join(TEMP_OUTPUT_DIR, "gallery"), exist_ok=True)
        os.makedirs(os.path.join(TEMP_OUTPUT_DIR, "downloads"), exist_ok=True)
        print(f"Ensured temporary subdirectories exist in {TEMP_OUTPUT_DIR}")
    except Exception as e_mkdir: print(f"Error creating subdirectories in {TEMP_OUTPUT_DIR}: {e_mkdir}")

if __name__ == "__main__":
    cleanup_temp_files(); print("Launching Gradio App..."); iface.launch()