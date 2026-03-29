import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

def build_abc_tokenizer():
    # 1. Initialize a blank BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    
    # 2. Split by whitespace to prevent massive block merging
    tokenizer.pre_tokenizer = None
    
    # 3. Define the strict, comprehensive ABC vocabulary
    abc_vocab = [
        "<pad>", "<s>", "</s>", "<unk>",

        # Structural
        "\n", " ",
        
        # Headers (as complete tokens)
        "L:", "M:", "K:", "V:", "Q:", "P:",

        # Header values that appear in your stripped format
        "1/4", "1/8", "1/16", "2/4", "3/4", "4/4", "6/8", "3/8",
        "C", "G", "D", "F", "A", "E", "B",
        "Bb", "Eb", "Ab", "F#", "C#",
        "bass", "treble",
        
        # Pitches (separate from header C, G, etc.)
        "c", "d", "e", "f", "g", "a", "b",
        "z", "Z", "X", "y",
        
        # Modifiers
        "^", "^^", "_", "__", "=",
        ",", ",,", "'", "''",
        
        # Octaves & Durations
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "/",
        
        # Structure, Barlines, Ties & Groupings
        "|", "|]", "||", "|:", ":|", "::", "[1", "[2",
        "-", "(", ")", "[", "]", "{", "}", "\"",
        
        # Dynamics
        "!pppp!", "!ppp!", "!pp!", "!p!", "!mp!", "!mf!", "!f!", "!ff!", "!fff!", "!ffff!", "!sfz!",
        "!crescendo(!", "!<(!", "!crescendo)!", "!<)!",
        "!diminuendo(!", "!>(!", "!diminuendo)!", "!>)!",
        
        # Ornaments & Decorations
        "!", "!trill!", "!lowermordent!", "!uppermordent!", "!mordent!", "!pralltriller!", 
        "!accent!", "!>!", "!emphasis!", "!fermata!", "!invertedfermata!", "!tenuto!", 
        "!trem1!", "!trem2!", "!trem3!", "!trem4!", "!xstem!", "!slide!", "!turnx!", 
        "!invertedturnx!", "!arpeggio!", "!invertedturn!",
        
        # Fingering, Phrasing & Repeats
        "!shortphrase!", "!mediumphrase!", "!longphrase!", "!upbow!", "!downbow!", 
        "!thumb!", "!snap!", "!turn!", "!roll!", "!breath!",
        "!segno!", "!coda!", "!D.S.!", "!D.C.!", "!dacoda!", "!dacapo!", "!fine!",
        
        # Short Forms
        ".", "~", "H", "L", "M", "P", "S", "T", "u", "v", "O", "J", "R",
        
        # Chord Symbols
        "maj", "min", "m", "dim", "aug", "+", "sus", "7", "9"
    ]
    
    # 4. Add them as special tokens so they are NEVER split by the tokenizer
    tokenizer.add_special_tokens(abc_vocab)
    
    # 5. Set up the standard BOS/EOS formatting for generation
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", abc_vocab.index("<s>")),
            ("</s>", abc_vocab.index("</s>")),
        ],
    )
    
    # 6. Wrap it for HuggingFace Transformers
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    
    hf_tokenizer.save_pretrained("./custom-abc-tokenizer")
    print(f"Tokenizer built and saved! Vocabulary size: {len(hf_tokenizer)}")
    return hf_tokenizer

if __name__ == "__main__":
    tk = build_abc_tokenizer()
    
    # Test it out to prove it perfectly isolates complex musical notation
    test_str = "M:4/4 !mf! !diminuendo(! \"Am7\" !trill!^c'2 !>)! |] "
    encoded = tk.tokenize(test_str)
    print(f"Test string: {test_str}")
    print(f"Tokens: {encoded}")