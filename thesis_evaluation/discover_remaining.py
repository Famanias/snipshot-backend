#!/usr/bin/env python3
"""
Discover texts in remaining test images (complex2, complex3).
With retry logic and delays for Groq API rate limits.
"""

import os
import sys
import json
import asyncio
import time
import numpy as np
from pathlib import Path
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def discover_remaining():
    from manga_translator import Config
    from manga_translator.manga_translator import MangaTranslator
    from manga_translator.utils import Context
    
    test_images_dir = os.path.join(str(project_root), "test images")
    
    # Only the remaining images
    remaining = [
        ("complex", os.path.join(test_images_dir, "complex", "test-image-complex2.jpg")),
        ("complex", os.path.join(test_images_dir, "complex", "test-image-complex3.png")),
    ]
    
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
    
    for category, img_path in remaining:
        img_file = os.path.basename(img_path)
        print(f"\n{'='*60}")
        print(f"Processing: {category}/{img_file}")
        print(f"{'='*60}")
        
        # Wait before each image to avoid rate limits
        print("  Waiting 30s for API rate limit...")
        await asyncio.sleep(30)
        
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
            
            # Translation with retry
            translated_texts = []
            text_regions_data = []
            if hasattr(ctx, 'text_regions') and ctx.text_regions:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        ctx.text_regions = await translator._run_text_translation(config, ctx)
                        for region in ctx.text_regions:
                            if hasattr(region, 'translation') and region.translation:
                                translated_texts.append(region.translation)
                                source = region.text if hasattr(region, 'text') else ""
                                print(f"    Translation: '{source}' -> '{region.translation}'")
                                text_regions_data.append({
                                    "source": source,
                                    "translation": region.translation
                                })
                        break
                    except Exception as e:
                        print(f"  Translation attempt {attempt+1} failed: {e}")
                        if attempt < max_retries - 1:
                            wait = 60 * (attempt + 1)
                            print(f"  Waiting {wait}s before retry...")
                            await asyncio.sleep(wait)
            
            results[f"{category}/{img_file}"] = {
                "category": category,
                "filename": img_file,
                "path": img_path,
                "resolution": list(img.size),
                "detected_texts": detected_texts,
                "confidences": confidences,
                "translated_texts": translated_texts,
                "text_regions": text_regions_data
            }
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    output_path = os.path.join(str(project_root), "thesis_evaluation", "discovered_texts_remaining.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n\nResults saved to: {output_path}")
    
    for key, data in results.items():
        print(f"  {key}: {len(data['detected_texts'])} texts, {len(data['translated_texts'])} translations")


if __name__ == "__main__":
    asyncio.run(discover_remaining())
