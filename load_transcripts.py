# load_transcripts.py
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube # Import the new library
import os
import re # Import regular expressions for sanitizing

# Manually curated list - replace with actual Huberman IDs if you changed them
VIDEO_IDS = [
    "jgaoLdS82vw", 
    "f4cdu-QiKHo", 
    "VnEy78RL2YY",
    "5--yogtN6oM", 
    "wFucddupQlk", 
    "Nmo4bxfFzM0", 
    "c9JmHOUp6VU",  
    # "PaN1wZwnYpA", # Example: Stress
    # "HcTlTMxoNCA", # Example: Goal Setting
    # "szqPAPKE5tQ"  # Example: Optimize Workspace
    # Using 10 example IDs
]

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def sanitize_filename(title):
    """Removes or replaces characters invalid for filenames."""
    # Remove most invalid characters (Windows list primarily)
    sanitized = re.sub(r'[\\/*?:"<>|]', "", title)
    # Replace spaces with underscores (optional, makes it cleaner)
    sanitized = sanitized.replace(" ", "_")
    # Shorten long filenames (e.g., to max 100 chars) to prevent OS issues
    max_len = 100
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len]
    # Ensure filename isn't empty after sanitization
    if not sanitized:
        return "untitled_video"
    return sanitized

def fetch_and_save_transcript(video_id):
    try:
        print(f"Processing Video ID: {video_id}")

        # --- Get Video Title using pytube ---
        try:
            yt = YouTube(f'https://www.youtube.com/watch?v={video_id}')
            video_title = yt.title
            sanitized_title = sanitize_filename(video_title)
            print(f"  Title: '{video_title}' -> Sanitized: '{sanitized_title}'")
        except Exception as e_title:
            print(f"  Error fetching title for {video_id}: {e_title}")
            print("  Using video ID as fallback filename.")
            sanitized_title = video_id # Fallback to ID if title fails
        # --- End Get Title ---

        print(f"  Attempting to fetch transcript...")
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        transcript = None
        try:
            # Try finding a manually created English transcript first
            transcript = transcript_list.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
            print(f"  Found manual English transcript.")
        except:
            try:
            # If no manual transcript, try finding a generated English transcript
                transcript = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
                print(f"  Found generated English transcript.")
            except Exception as e_gen:
                 print(f"  Could not find manual or generated English transcript for {video_id}: {e_gen}")
                 # Optionally try any language as fallback (might not be useful for RAG)
                 # try:
                 #    transcript = transcript_list.find_generated_transcript(transcript_list.languages)
                 #    print(f"  Warning: Using first available generated transcript ({transcript.language}) for {video_id}")
                 # except:
                 #    raise Exception("No suitable transcript found.") # Re-raise if nothing found
                 return False # Skip if no English transcript found

        if transcript is None:
             print(f"  Skipping {video_id} as no English transcript was ultimately found.")
             return False

        segments = transcript.fetch()
        if not segments:
            print(f"  Warning: Fetched segments list is empty for {video_id}")
            full_transcript = "" # Handle empty transcripts
        else:
             # Use attribute access '.text' which worked before
             full_transcript = " ".join([segment.text for segment in segments])

        # Use the sanitized title for the filename
        file_path = os.path.join(DATA_DIR, f"{sanitized_title}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_transcript)
        print(f"  Successfully saved transcript to '{file_path}'")
        return True

    except Exception as e:
        # Print the specific error for the problematic video ID
        print(f"-> Error processing {video_id}: {type(e).__name__} - {e}")
        return False

if __name__ == "__main__":
    print(f"Fetching transcripts for {len(VIDEO_IDS)} videos...")
    success_count = 0
    failure_count = 0
    for video_id in VIDEO_IDS:
        if fetch_and_save_transcript(video_id):
            success_count += 1
        else:
            failure_count += 1
        print("-" * 20) # Separator between videos
    print(f"Finished.")
    print(f"Successfully fetched: {success_count}/{len(VIDEO_IDS)}")
    print(f"Failures/Skipped: {failure_count}/{len(VIDEO_IDS)}")