import os
import time
from card_formatter import CardFormatter
from PIL import Image

# תיקיית הפלט של YOLO
OUTPUT_DIR = r"C:\Poker_3.11.25\output"
# תיקיית תמונות הקלפים
CARDS_DIR = r"C:\Poker_3.11.25\cards_images"
# תיקייה להצגת קלפים (ל-OBS)
DISPLAY_DIR = r"C:\Poker_3.11.25\display"

os.makedirs(DISPLAY_DIR, exist_ok=True)

def read_cards(file_path):
    """קורא את תוכן הקובץ ומחזיר רשימת קלפים"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            if not line:
                return []
            return [c.strip() for c in line.split(",")]
    except:
        return []

def create_display_image(player_num, cards):
    """יוצר תמונה אחת עם שני הקלפים (להצגה ב־OBS)"""
    if not cards:
        return

    card_files = [CardFormatter.to_filename(c) for c in cards]
    images = []
    for file in card_files:
        path = os.path.join(CARDS_DIR, file)
        if os.path.exists(path):
            images.append(Image.open(path))

    if not images:
        return

    # מחברים שתי תמונות זו לצד זו
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)
    combined = Image.new("RGBA", (total_width, max_height))

    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    out_path = os.path.join(DISPLAY_DIR, f"player{player_num}_cards.png")
    combined.save(out_path)
    print(f"עודכן קלפים לשחקן {player_num}: {cards} -> {out_path}")

def main():
    print("🎴 מציג קלפים בזמן אמת... (Ctrl+C לעצירה)")
    while True:
        for i in range(1, 9):
            player_file = os.path.join(OUTPUT_DIR, f"player{i}.txt")
            if os.path.exists(player_file):
                cards = read_cards(player_file)
                create_display_image(i, cards)
        time.sleep(2)

if __name__ == "__main__":
    main()
