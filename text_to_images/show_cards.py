"""
Continuous Text-to-Image Card Visualizer for Poker
Reads all .txt files in output/ and generates/updates images in card_display/
"""

import os
import time
from pathlib import Path
from PIL import Image

# ===== CONFIG =====
BASE_DIR = Path(r"C:\Poker_3.11.25\text_to_images")  # כל התיקיות תחת text_to_images
INPUT_DIR = Path(r"C:\Poker_3.11.25\output")         # קבצי הטקסט של השחקנים
CARD_IMG_DIR = BASE_DIR / "card_visualizer"          # תיקיית קלפים
OUTPUT_DIR = BASE_DIR / "card_display"               # תיקיית תמונות המשולבות
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CARD_SIZE = (150, 210)  # גודל קלף בפיקסלים
SPACING = 10            # רווח בין קלפים
SLEEP_TIME = 2          # שניות בין בדיקות חוזרות

# לשמירת זמן השינוי האחרון של קבצים
last_mtime = {}

# ===== FUNCTIONS =====
def clean_card_string(card: str) -> str:
    """Remove extra spaces and normalize symbols"""
    return card.strip().replace("❤", "♥")

def card_to_image_name(card_str: str) -> str:
    """Convert card like 'J♠' or 'Q♥' -> 'JS.png' or 'QH.png'"""
    suit_map = {'♠': 'S', '♥': 'H', '♦': 'D', '♣': 'C', '❤': 'H'}
    card_str = clean_card_string(card_str)
    if card_str.startswith('10'):
        rank = '10'
        suit = suit_map.get(card_str[2], card_str[2])
    else:
        rank = card_str[0]
        suit = suit_map.get(card_str[1], card_str[1])
    return f"{rank}{suit}.png"

def generate_player_image(txt_path: Path):
    """Generate combined image for a single player"""
    player_name = txt_path.stem
    # בדיקה אם הקובץ השתנה
    mtime = txt_path.stat().st_mtime
    if last_mtime.get(player_name) == mtime:
        return  # לא השתנה
    last_mtime[player_name] = mtime

    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    if not content:
        print(f"No cards found in {player_name}")
        return

    card_codes = [c.strip() for c in content.split(",")]
    card_images = []

    for code in card_codes:
        img_name = card_to_image_name(code)
        img_path = CARD_IMG_DIR / img_name
        if not img_path.exists():
            print(f"WARNING: Image not found for card {code} -> {img_path}")
            continue
        card_img = Image.open(img_path).resize(CARD_SIZE)
        card_images.append(card_img)

    if not card_images:
        print(f"No valid card images for {player_name}")
        return

    # שילוב תמונות אופקית
    total_width = len(card_images) * CARD_SIZE[0] + (len(card_images) - 1) * SPACING
    total_height = CARD_SIZE[1]
    combined = Image.new('RGBA', (total_width, total_height), (0,0,0,0))
    x_offset = 0
    for img in card_images:
        combined.paste(img, (x_offset, 0), mask=img)
        x_offset += CARD_SIZE[0] + SPACING

    output_file = OUTPUT_DIR / f"{player_name}.png"
    combined.save(output_file)
    print(f"Updated image for {player_name}: {output_file}")

# ===== MAIN LOOP =====
print("Starting continuous text-to-image updater...")

while True:
    txt_files = list(INPUT_DIR.glob("*.txt"))
    for txt_file in txt_files:
        generate_player_image(txt_file)
    time.sleep(SLEEP_TIME)
