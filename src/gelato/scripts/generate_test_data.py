import os
from pathlib import Path
import shutil
from gelato.data.converter import convert_xml_to_abc
from gelato.data.renderer import Renderer
from gelato.data.canonicalize import canonicalize_abc

def main():
    # 1. Create dummy MusicXML
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1">
      <part-name>Music</part-name>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key>
          <fifths>0</fifths>
        </key>
        <time>
          <beats>4</beats>
          <beat-type>4</beat-type>
        </time>
        <clef>
          <sign>G</sign>
          <line>2</line>
        </clef>
      </attributes>
      <note>
        <pitch>
          <step>C</step>
          <octave>4</octave>
        </pitch>
        <duration>4</duration>
        <type>whole</type>
      </note>
    </measure>
  </part>
</score-partwise>
"""
    raw_dir = Path("data/raw_test")
    raw_dir.mkdir(parents=True, exist_ok=True)
    xml_path = raw_dir / "test_sample.musicxml"
    with open(xml_path, "w") as f:
        f.write(xml_content)
        
    print(f"Created {xml_path}")
    
    # 2. Convert to ABC
    abc_dir = Path("data/abc_test")
    abc_file = convert_xml_to_abc(xml_path, abc_dir / "test_sample.abc")
    print(f"Converted to {abc_file}")
    
    # 3. Canonicalize
    canonicalize_abc(abc_file)
    print("Canonicalized ABC.")
    
    # 4. Render
    renderer_out = Path("data/render_test")
    renderer = Renderer(renderer_out)
    svgs = renderer.render_abc_to_svg(abc_file)
    print(f"Rendered SVGs: {svgs}")
    
    # 5. Process Images
    # Create final sample structure
    processed_dir = Path("data/processed_test/sample_001")
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "patches").mkdir()
    
    # Copy ABC
    shutil.copy(abc_file, processed_dir / "label.abc")
    
    # Process first SVG
    if svgs:
        png = renderer.convert_svg_to_png(svgs[0])
        segments = renderer.process_image_for_model(png)
        # We likely only have 1 segment for this tiny file.
        # Slice it.
        patches = renderer.slice_segment(segments[0])
        for i, patch in enumerate(patches):
            patch.save(processed_dir / "patches" / f"{i}.png")
        print(f"Saved {len(patches)} patches to {processed_dir}")
        
if __name__ == "__main__":
    main()
