import sys
from pathlib import Path
import torch

try:
    from gelato.data.converter import convert_xml_to_abc
    from gelato.data.canonicalize import canonicalize_abc
    from gelato.data.renderer import Renderer
    from gelato.model.modeling import GelatoModel, GelatoConfig
    from gelato.utils.tokenizer import get_tokenizer
    print("[PASS] Imports successful.")
except ImportError as e:
    print(f"[FAIL] Imports failed: {e}")
    sys.exit(1)

def test_external_tools():
    import subprocess
    # Check abcm2ps
    try:
        subprocess.run(["abcm2ps", "-V"], check=True, capture_output=True)
        print("[PASS] abcm2ps found.")
    except Exception as e:
        print(f"[FAIL] abcm2ps check failed: {e}")

    # Check convert
    try:
        subprocess.run(["convert", "--version"], check=True, capture_output=True)
        print("[PASS] ImageMagick convert found.")
    except Exception as e:
        print(f"[FAIL] ImageMagick convert check failed: {e}")

def test_model_load():
    try:
        # Use smaller dummy config for speed check
        config = GelatoConfig(
            vision_model_name="google/siglip-so400m-patch14-384", # Smaller available one for test if needed, or stick to default
            # Actually let's just instantiate the classes without loading weights if possible, or load generic
            # But the user might not have networking allowed for huggingface if restricted? 
            # Assuming standard env.
        )
        # Just check classes instantiate if we mock
        # Real load might take time / memory.
        print("[INFO] Skipping full model load in quick test, checking class structure.")
        
        # Test Resampler
        from gelato.model.resampler import PerceiverResampler
        resampler = PerceiverResampler(dim=128, input_dim=128, depth=1)
        x = torch.randn(2, 10, 128)
        out = resampler(x)
        assert out.shape == (2, 256, 128)
        print("[PASS] PerceiverResampler forward pass shape correct.")
        
    except Exception as e:
        print(f"[FAIL] Model components test failed: {e}")

if __name__ == "__main__":
    test_external_tools()
    test_model_load()
