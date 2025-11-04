"""
Poker Hand Detection System using YOLO11
Monitors multiple players' phone cameras via ADB/scrcpy and detects poker hands.
"""

import os
import sys
import time
import subprocess
import tempfile
import json
from pathlib import Path
from typing import List, Optional, Dict
import logging

# Add YOLO11 detection module to path
yolo_path = Path(__file__).parent / "yolo11-poker-hand-detection-and-analysis-main"
sys.path.insert(0, str(yolo_path))

from ultralytics import YOLO
from detect_cards import detect_cards

# ==================== CONFIGURATION ====================
# Number of players to monitor (1-8)
NUM_PLAYERS = 8

# Delay in seconds between consecutive detections
DETECTION_DELAY = 5

# Paths (adjust these if needed)
SCRCPY_PATH = Path(r"C:\Poker_3.11.25\scrcpy")
YOLO_WEIGHTS_PATH = Path(r"C:\Poker_3.11.25\yolo11-poker-hand-detection-and-analysis-main\weights\poker_best.pt")
OUTPUT_DIR = Path(r"C:\Poker_3.11.25\output")
DEVICE_MAPPING_FILE = Path(r"C:\Poker_3.11.25\device_mapping.json")

# ADB executable path (within scrcpy directory)
ADB_EXE = SCRCPY_PATH / "adb.exe"

# YOLO detection confidence threshold
CONFIDENCE_THRESHOLD = 0.3  # Lowered for better detection

# Logging configuration
LOG_LEVEL = logging.INFO
# ========================================================

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class CardFormatter:
    """Converts card codes (e.g., 'JH') to display format (e.g., 'J♥')."""
    
    # Suit symbols mapping
    SUIT_SYMBOLS = {
        'C': '♣',  # Clubs
        'D': '♦',  # Diamonds
        'H': '♥',  # Hearts
        'S': '♠'   # Spades
    }
    
    # Rank mapping (to handle face cards)
    RANK_MAP = {
        'J': 'J',
        'Q': 'Q',
        'K': 'K',
        'A': 'A'
    }
    
    @staticmethod
    def format_card(card_code: str) -> str:
        """
        Convert card code to display format.
        
        Args:
            card_code: Card in dataset format (e.g., 'JH', '10D', '2C')
        
        Returns:
            Formatted card string (e.g., 'J♥', '10♦', '2♣')
        """
        if len(card_code) < 2:
            return card_code
        
        # Handle 10 (two-character rank)
        if card_code.startswith('10'):
            rank = '10'
            suit = card_code[2]
        else:
            rank = card_code[0]
            suit = card_code[1]
        
        suit_symbol = CardFormatter.SUIT_SYMBOLS.get(suit, suit)
        return f"{rank}{suit_symbol}"


class DeviceManager:
    """Manages device serial number to player name mapping."""
    
    def __init__(self, mapping_file: Path):
        """
        Initialize device manager.
        
        Args:
            mapping_file: Path to JSON file storing device mappings
        """
        self.mapping_file = mapping_file
        self.device_to_player: Dict[str, str] = {}
        self.load_mappings()
    
    def load_mappings(self):
        """Load device mappings from JSON file."""
        try:
            if self.mapping_file.exists():
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    self.device_to_player = json.load(f)
                logger.info(f"Loaded {len(self.device_to_player)} device mapping(s) from {self.mapping_file}")
            else:
                logger.info("No existing device mappings found. New mappings will be created.")
        except Exception as e:
            logger.warning(f"Error loading device mappings: {e}. Starting with empty mappings.")
            self.device_to_player = {}
    
    def save_mappings(self):
        """Save device mappings to JSON file."""
        try:
            with open(self.mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.device_to_player, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved device mappings to {self.mapping_file}")
        except Exception as e:
            logger.error(f"Error saving device mappings: {e}")
    
    def get_player_name(self, device_serial: str) -> Optional[str]:
        """
        Get player name for a device serial number.
        
        Args:
            device_serial: Device serial number
        
        Returns:
            Player name if mapped, None otherwise
        """
        return self.device_to_player.get(device_serial)
    
    def register_device(self, device_serial: str, player_name: str):
        """
        Register a new device with a player name.
        
        Args:
            device_serial: Device serial number
            player_name: Player name
        """
        self.device_to_player[device_serial] = player_name
        self.save_mappings()
        logger.info(f"Registered device {device_serial} as player: {player_name}")
    
    def prompt_for_player_name(self, device_serial: str) -> str:
        """
        Prompt user for player name for a new device.
        
        Args:
            device_serial: Device serial number
        
        Returns:
            Player name entered by user
        """
        print(f"\nNew device detected: {device_serial}")
        while True:
            player_name = input("Enter a name for this player: ").strip()
            if player_name:
                # Sanitize filename - remove invalid characters
                invalid_chars = '<>:"/\\|?*'
                sanitized_name = ''.join(c for c in player_name if c not in invalid_chars)
                if sanitized_name != player_name:
                    logger.warning(f"Removed invalid characters from player name: {player_name} -> {sanitized_name}")
                    player_name = sanitized_name
                
                if player_name:
                    self.register_device(device_serial, player_name)
                    return player_name
            print("Please enter a valid player name.")


class ADBManager:
    """Manages ADB connections and operations for multiple devices."""
    
    def __init__(self, adb_path: Path):
        """
        Initialize ADB manager.
        
        Args:
            adb_path: Path to adb.exe
        """
        self.adb_path = adb_path
        if not adb_path.exists():
            raise FileNotFoundError(f"ADB not found at {adb_path}")
    
    def get_connected_devices(self) -> List[str]:
        """
        Get list of connected device IDs.
        
        Returns:
            List of device serial numbers
        """
        try:
            result = subprocess.run(
                [str(self.adb_path), "devices"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                logger.error(f"Failed to list devices: {result.stderr}")
                return []
            
            devices = []
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if line.strip() and '\tdevice' in line:
                    device_id = line.split('\t')[0].strip()
                    if device_id:
                        devices.append(device_id)
            
            return devices
        except subprocess.TimeoutExpired:
            logger.error("ADB devices command timed out")
            return []
        except Exception as e:
            logger.error(f"Error getting devices: {e}")
            return []
    
    def capture_from_camera(self, device_id: str, output_path: Path) -> bool:
        """
        Capture from device camera preview (screen capture when camera app is open).
        Note: This method captures the screen while camera preview is visible.
        For best results, manually open the camera app on the device first.
        
        Args:
            device_id: Device serial number
            output_path: Path to save captured image
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # First, try to ensure camera app is open (or use screen if already open)
            # We'll capture the screen which should show camera preview if camera app is open
            logger.info(f"Capturing from device {device_id} (ensure camera app is open on device)")
            
            # Use screen capture - this will capture camera preview if camera app is open
            # This is more reliable than trying to trigger camera programmatically
            return self.capture_screenshot(device_id, output_path)
            
        except Exception as e:
            logger.warning(f"Error capturing from camera {device_id}: {e}")
            return False
    
    def capture_screenshot(self, device_id: str, output_path: Path) -> bool:
        """
        Capture screenshot from a specific device (fallback method).
        
        Args:
            device_id: Device serial number
            output_path: Path to save screenshot
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use exec-out to get binary screenshot data
            result = subprocess.run(
                [str(self.adb_path), "-s", device_id, "exec-out", "screencap", "-p"],
                capture_output=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logger.warning(f"Failed to capture screenshot from {device_id}: {result.stderr}")
                return False
            
            # Write binary data to file
            with open(output_path, 'wb') as f:
                f.write(result.stdout)
            
            return output_path.exists() and output_path.stat().st_size > 0
        except subprocess.TimeoutExpired:
            logger.warning(f"Screenshot capture timed out for device {device_id}")
            return False
        except Exception as e:
            logger.warning(f"Error capturing screenshot from {device_id}: {e}")
            return False


class PokerHandDetector:
    """Main class for poker hand detection system."""
    
    def __init__(
        self,
        num_players: int,
        detection_delay: float,
        adb_path: Path,
        weights_path: Path,
        output_dir: Path,
        device_mapping_file: Path,
        confidence: float = 0.5
    ):
        """
        Initialize poker hand detector.
        
        Args:
            num_players: Maximum number of players to monitor
            detection_delay: Delay between detections in seconds
            adb_path: Path to adb.exe
            weights_path: Path to YOLO weights file
            output_dir: Directory to save output files
            device_mapping_file: Path to device mapping JSON file
            confidence: YOLO confidence threshold
        """
        self.num_players = num_players
        self.detection_delay = detection_delay
        self.adb_manager = ADBManager(adb_path)
        self.device_manager = DeviceManager(device_mapping_file)
        self.weights_path = weights_path
        self.output_dir = output_dir
        self.confidence = confidence
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate weights file
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found at {weights_path}")
        
        # Load YOLO model once for reuse
        self.model = YOLO(str(weights_path), task='detect')
        logger.info("YOLO model loaded successfully")
        
        # Create temporary directory for captured images
        self.temp_dir = Path(tempfile.mkdtemp(prefix="poker_detection_"))
        logger.info(f"Using temporary directory: {self.temp_dir}")
    
    def detect_cards_in_image(self, image_path: Path) -> Optional[List[str]]:
        """
        Detect cards in an image using YOLO11.
        
        Args:
            image_path: Path to image file
        
        Returns:
            List of detected card codes, or None if detection failed
        """
        try:
            # Use YOLO model directly for better control
            results = self.model.predict(
                str(image_path),
                conf=self.confidence,
                verbose=False
            )
            
            result = results[0]
            
            # Extract card names
            cards = []
            cards_names = []
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf >= self.confidence:
                        cls = int(box.cls[0])
                        card_name = result.names[cls]
                        if card_name not in cards_names:
                            cards_names.append(card_name)
                            # Get left position for sorting
                            x1 = float(box.xyxy[0][0])
                            cards.append((x1, card_name))
                
                # Sort by left position
                cards.sort(key=lambda x: x[0])
                card_names = [card[1] for card in cards]
            else:
                # Fallback to original detect_cards function
                card_names = detect_cards(str(image_path), str(self.weights_path), conf=self.confidence)
            
            return card_names
            
        except Exception as e:
            logger.error(f"Error detecting cards in {image_path}: {e}")
            return None
    
    def validate_and_format_cards(self, cards: List[str]) -> Optional[str]:
        """
        Validate that exactly 2 cards are detected and format them.
        
        Args:
            cards: List of detected card codes
        
        Returns:
            Formatted card string (e.g., "J♦, A♥") or None if invalid
        """
        if len(cards) != 2:
            return None
        
        formatted_cards = [CardFormatter.format_card(card) for card in cards]
        return ", ".join(formatted_cards)
    
    def save_player_result(self, player_name: str, card_string: str):
        """
        Save detected cards to player's output file.
        
        Args:
            player_name: Player name
            card_string: Formatted card string
        """
        output_file = self.output_dir / f"{player_name}.txt"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(card_string)
            logger.info(f"Saved result for {player_name}: {card_string}")
        except Exception as e:
            logger.error(f"Error saving result for {player_name}: {e}")
    
    def process_player(self, device_id: str, player_name: str) -> bool:
        """
        Process a single player's device.
        
        Args:
            device_id: Device serial number
            player_name: Player name
        
        Returns:
            True if successful, False otherwise
        """
        image_path = self.temp_dir / f"{player_name}_capture.png"
        
        # Capture from camera (captures screen with camera preview if camera app is open)
        if not self.adb_manager.capture_from_camera(device_id, image_path):
            logger.warning(f"Failed to capture image for {player_name} (device {device_id})")
            return False
        
        # Detect cards
        cards = self.detect_cards_in_image(image_path)
        if cards is None:
            logger.warning(f"Card detection failed for {player_name}")
            return False
        
        # Log detection results
        if len(cards) == 0:
            logger.info(f"{player_name}: No cards detected")
        else:
            logger.info(f"{player_name}: Detected {len(cards)} card(s): {cards}")
        
        # Validate and format
        formatted_result = self.validate_and_format_cards(cards)
        if formatted_result is None:
            logger.info(f"{player_name}: Invalid detection - {len(cards)} cards detected (expected exactly 2)")
            return False
        
        # Save result
        self.save_player_result(player_name, formatted_result)
        logger.info(f"{player_name}: Successfully detected and saved: {formatted_result}")
        
        return True
    
    def run(self):
        """Main detection loop."""
        logger.info("=" * 60)
        logger.info("Starting Poker Hand Detection System")
        logger.info("=" * 60)
        logger.info(f"Monitoring up to {self.num_players} players")
        logger.info(f"Detection delay: {self.detection_delay} seconds")
        logger.info(f"Confidence threshold: {self.confidence}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("")
        logger.info("IMPORTANT: For best results:")
        logger.info("  1. Open the camera app on each device")
        logger.info("  2. Point camera at the poker cards")
        logger.info("  3. Ensure cards are clearly visible in camera preview")
        logger.info("  4. Results will be saved to: output/[PlayerName].txt")
        logger.info("")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        try:
            while True:
                # Get connected devices
                devices = self.adb_manager.get_connected_devices()
                
                if not devices:
                    logger.warning("No devices connected. Waiting...")
                    time.sleep(self.detection_delay)
                    continue
                
                # Limit to configured number of players
                active_devices = devices[:self.num_players]
                logger.info(f"Found {len(active_devices)} device(s) (monitoring up to {self.num_players})")
                
                # Process each device
                for device_id in active_devices:
                    try:
                        # Get or prompt for player name
                        player_name = self.device_manager.get_player_name(device_id)
                        
                        if player_name is None:
                            # New device - prompt for name
                            player_name = self.device_manager.prompt_for_player_name(device_id)
                        
                        # Process the device with the player name
                        self.process_player(device_id, player_name)
                    except Exception as e:
                        logger.error(f"Error processing device {device_id}: {e}")
                        continue
                
                # Wait before next detection cycle
                logger.debug(f"Waiting {self.detection_delay} seconds before next detection...")
                time.sleep(self.detection_delay)
                
        except KeyboardInterrupt:
            logger.info("Detection stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
        finally:
            # Cleanup temporary files
            try:
                import shutil
                if self.temp_dir.exists():
                    shutil.rmtree(self.temp_dir)
                    logger.info("Cleaned up temporary files")
            except Exception as e:
                logger.warning(f"Error cleaning up temporary files: {e}")


def main():
    """Main entry point."""
    try:
        # Create detector instance
        detector = PokerHandDetector(
            num_players=NUM_PLAYERS,
            detection_delay=DETECTION_DELAY,
            adb_path=ADB_EXE,
            weights_path=YOLO_WEIGHTS_PATH,
            output_dir=OUTPUT_DIR,
            device_mapping_file=DEVICE_MAPPING_FILE,
            confidence=CONFIDENCE_THRESHOLD
        )
        
        # Run detection
        detector.run()
        
    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

