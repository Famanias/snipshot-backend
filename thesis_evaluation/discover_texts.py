#!/usr/bin/env python3
"""
Discover texts in test images by running the pipeline once.
This helps build ground truth data for benchmark evaluation.
"""

import os
import sys
import json
import asyncio
import numpy as np
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def discover_texts():
    from manga_translator import Config
    from manga_translator.manga_translator import MangaTranslator
    from manga_translator.utils import Context
    
    # All test images
    test_images_dir = os.path.join(str(project_root), "test images")
    categories = {
        "easy": os.path.join(test_images_dir, "easy"),
        "medium": os.path.join(test_images_dir, "medium"),
        "complex": os.path.join(test_images_dir, "complex"),
    }
    
    config = Config(**{
        "detector": {"detector": "default", "detection_size": 1536, "box_threshold": 0.7},
        "ocr": {"ocr": "48px"},
        "translator": {"translator": "groq", "target_lang": "ENG"},
        "inpainter": {"inpainter": "default"},
        "render": {"renderer": "default"}
    })
    
    translator = MangaTranslator({
        'use_gpu': True,
        'kernel_size': 3,
        'verbose': False
    })
    
    results = {}
    
    for category, folder in categories.items():
        for img_file in sorted(os.listdir(folder)):
            img_path = os.path.join(folder, img_file)
            print(f"\n{'='*60}")
            print(f"Processing: {category}/{img_file}")
            print(f"{'='*60}")
            
            try:
                img = Image.open(img_path).convert('RGB')
                print(f"  Resolution: {img.size}")
                
                ctx = Context()
                ctx.input = img
                ctx.result = None
                ctx.img_rgb = np.array(img)
                
                # Detection
                detection_result = await translator._run_detection(config, ctx)
                ctx.textlines = detection_result[0] if isinstance(detection_result, tuple) else detection_result
                ctx.mask_raw = detection_result[1] if isinstance(detection_result, tuple) and len(detection_result) > 1 else None
                print(f"  Detected {len(ctx.textlines) if ctx.textlines else 0} text regions")
                
                # OCR
                detected_texts = []
                confidences = []
                if ctx.textlines:
                    ctx.textlines = await translator._run_ocr(config, ctx)
                    for tl in ctx.textlines:
                        if hasattr(tl, 'text') and tl.text:
                            detected_texts.append(tl.text)
                            conf = float(tl.prob) if hasattr(tl, 'prob') else 1.0
                            confidences.append(conf)
                            print(f"    Text: '{tl.text}' (conf: {conf:.4f})")
                
                # Textline merge
                if ctx.textlines:
                    ctx.text_regions = await translator._run_textline_merge(config, ctx)
                    ctx.mask = await translator._run_mask_refinement(config, ctx)
                
                # Translation
                translated_texts = []
                if hasattr(ctx, 'text_regions') and ctx.text_regions:
                    ctx.text_regions = await translator._run_text_translation(config, ctx)
                    for region in ctx.text_regions:
                        if hasattr(region, 'translation') and region.translation:
                            translated_texts.append(region.translation)
                            # Get source text
                            source = region.text if hasattr(region, 'text') else ""
                            print(f"    Translation: '{source}' -> '{region.translation}'")
                
                results[f"{category}/{img_file}"] = {
                    "category": category,
                    "filename": img_file,
                    "path": img_path,
                    "resolution": list(img.size),
                    "detected_texts": detected_texts,
                    "confidences": confidences,
                    "translated_texts": translated_texts,
                    "text_regions": []
                }
                
                # Build text_regions and translations for ground truth
                if hasattr(ctx, 'text_regions') and ctx.text_regions:
                    for region in ctx.text_regions:
                        source = ""
                        if hasattr(region, 'text'):
                            source = region.text
                        translation = ""
                        if hasattr(region, 'translation'):
                            translation = region.translation
                        results[f"{category}/{img_file}"]["text_regions"].append({
                            "source": source,
                            "translation": translation
                        })
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    # Save results
    output_path = os.path.join(str(project_root), "thesis_evaluation", "discovered_texts.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n\nResults saved to: {output_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_texts = 0
    total_translations = 0
    for key, data in results.items():
        n_texts = len(data['detected_texts'])
        n_trans = len(data['translated_texts'])
        total_texts += n_texts
        total_translations += n_trans
        print(f"  {key}: {n_texts} texts, {n_trans} translations")
    print(f"  TOTAL: {total_texts} texts, {total_translations} translations")


if __name__ == "__main__":
    asyncio.run(discover_texts())
