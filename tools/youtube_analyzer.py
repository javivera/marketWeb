#!/usr/bin/env python3
"""
YouTube Video Transcript Analyzer with Gemini AI
Single script that extracts YouTube transcripts and analyzes them with AI
"""

import os
import re
import urllib.parse
from datetime import datetime
from pathlib import Path

def load_env_file():
    """Load environment variables from .env file in script directory"""
    script_dir = Path(__file__).parent
    env_file = script_dir / '.env'
    
    env_vars = {}
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"\'')
        print(f"✅ Loaded .env from: {env_file}")
    else:
        print(f"⚠️  .env not found at: {env_file}")
    
    return env_vars

def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    # Handle escaped URLs and clean them up
    url = urllib.parse.unquote(url)
    # Remove backslashes that might be escaping characters
    url = url.replace('\\', '')
    
    print(f"🔍 Cleaned URL: {url}")
    
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})',
        r'(?:youtube\.com.*v=)([0-9A-Za-z_-]{11})',
    ]
    
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            print(f"✅ Video ID found using pattern {i+1}: {video_id}")
            return video_id
    
    print(f"❌ No video ID found in URL: {url}")
    return None

def get_video_title(video_id):
    """Get YouTube video title"""
    try:
        import requests
        url = f"https://www.youtube.com/watch?v={video_id}"
        response = requests.get(url)
        
        # Extract title from HTML
        import re
        title_match = re.search(r'<title>(.+?) - YouTube</title>', response.text)
        if title_match:
            title = title_match.group(1)
            # Clean the title for filename
            title = re.sub(r'[<>:"/\\|?*]', '', title)  # Remove invalid filename chars
            title = title.strip()
            return title
    except Exception as e:
        print(f"⚠️  Could not fetch video title: {e}")
    
    return None

def get_transcript(video_url):
    """Get transcript from YouTube video"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        print("❌ Install: pip install youtube-transcript-api")
        return None
    
    video_id = extract_video_id(video_url)
    if not video_id:
        print("❌ Invalid YouTube URL")
        return None
    
    print(f"📹 Video ID: {video_id}")
    
    try:
        # Use the API pattern you specified
        ytt_api = YouTubeTranscriptApi()
        transcript_data = ytt_api.fetch(video_id)
        
        print("✅ Transcript fetched successfully")
        
    except Exception as e:
        print(f"❌ Failed to get transcript: {e}")
        print("💡 This video might not have captions available.")
        return None
    
    # Handle the response format
    if isinstance(transcript_data, list):
        # Standard format: list of segments
        full_transcript = ""
        for segment in transcript_data:
            if isinstance(segment, dict):
                text = segment.get('text', '').strip()
            else:
                text = str(segment).strip()
            if text:
                full_transcript += text + " "
    else:
        # If it's not a list, convert to string
        full_transcript = str(transcript_data)
    
    full_transcript = re.sub(r'\s+', ' ', full_transcript).strip()
    print(f"✅ Transcript extracted ({len(full_transcript)} characters)")
    return full_transcript

def list_gemini_models(api_key):
    """List available Gemini models"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        print("🤖 Available Gemini models:")
        models = genai.list_models()
        for model in models:
            print(f"   - {model.name}")
        
        return [model.name for model in models]
        
    except Exception as e:
        print(f"❌ Error listing models: {e}")
        return []

def analyze_with_gemini(transcript):
    """Analyze transcript with Gemini"""
    try:
        import google.generativeai as genai
    except ImportError:
        print("❌ Install: pip install google-generativeai")
        return None
    
    env_vars = load_env_file()
    api_key = env_vars.get('GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env file")
        return None
    
    try:
        genai.configure(api_key=api_key)
        
        # List available models
        available_models = list_gemini_models(api_key)
        
        # Choose model (prefer gemini-pro, fall back to first available)
        model_name = 'gemini-2.5-flash-lite'
        if available_models:
            
            # Check if gemini-pro is available
            if not any('gemini-pro' in model for model in available_models):
                model_name = available_models[0]
                print(f"⚠️  gemini-pro not found, using: {model_name}")
        
        print(f"🤖 Using model: {model_name}")
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""Analyze this market analysis YouTube video transcript and provide:

## Executive Summary
A comprehensive summary of the key market insights and analyst predictions (2-3 paragraphs)

## Stocks & Securities Mentioned
- List all stocks, ETFs, cryptocurrencies, or other securities discussed
- Include ticker symbols where mentioned
- Note any price targets or recommendations

## Market Analysis & Insights
- Key market trends identified
- Economic indicators discussed
- Sector analysis and themes
- Technical analysis points (support/resistance levels, chart patterns)

## Investment Recommendations
- Specific buy/sell/hold recommendations
- Risk assessments mentioned
- Time horizons for trades/investments
- Position sizing or allocation suggestions

## Key Market Events & Catalysts
- Upcoming earnings, events, or announcements
- Economic data releases mentioned
- Geopolitical factors affecting markets

## Notable Quotes & Predictions
- Important predictions or forecasts
- Specific price targets or timeframes
- Contrarian views or bold statements

## Risk Factors & Disclaimers
- Risks mentioned by the analyst
- Market uncertainties highlighted
- Any disclaimers about the analysis

## Overall Assessment
Your evaluation of the analysis quality, reasoning, and potential value for investors

---

Transcript:
{transcript}"""
        
        print("🤖 Analyzing with Gemini...")
        response = model.generate_content(prompt)
        
        if response.text:
            print("✅ Analysis completed")
            return response.text
        else:
            print("❌ No response from Gemini")
            return None
            
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return None

def save_analysis(video_url, transcript, analysis):
    """Save analysis to file"""
    video_id = extract_video_id(video_url)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Try to get video title for filename
    video_title = get_video_title(video_id)
    if video_title:
        # Use title + timestamp for filename
        filename = f"{video_title}_{timestamp}.md"
        print(f"📺 Video title: {video_title}")
    else:
        # Fallback to video ID if title fetch fails
        filename = f"{video_id}_{timestamp}.md"
    
    # Create extractions folder
    extractions_dir = Path(__file__).parent / "extractions"
    extractions_dir.mkdir(exist_ok=True)
    
    output_file = extractions_dir / filename
    
    # Create content
    content = f"""# YouTube Video Analysis
**Video ID:** {video_id}
**Title:** {video_title or 'Unknown'}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**URL:** {video_url}

---

{analysis}

---

## Original Transcript
{transcript}
"""
    
    # Save file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"💾 Analysis saved: {output_file}")
    return output_file

def main():
    """Main function"""
    import sys
    
    print("🎬 YouTube Transcript Analyzer with Gemini AI")
    print("=" * 50)
    
    # Get video URL from command line argument
    if len(sys.argv) != 2:
        print("Usage: python youtube_analyzer.py <youtube_url>")
        print("Example: python youtube_analyzer.py 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'")
        return
    
    video_url = sys.argv[1].strip()
    print(f"🔗 Processing URL: {video_url}")
    
    # Get transcript
    print("\n📝 Extracting transcript...")
    transcript = get_transcript(video_url)
    
    if not transcript:
        print("❌ Could not extract transcript")
        return
    
    # Analyze with Gemini
    print("\n🤖 Analyzing with Gemini...")
    analysis = analyze_with_gemini(transcript)
    
    if not analysis:
        print("❌ Could not analyze transcript")
        return
    
    # Save results
    print("\n💾 Saving results...")
    output_file = save_analysis(video_url, transcript, analysis)
    
    print(f"\n✅ Complete! Analysis saved to: {output_file}")

if __name__ == "__main__":
    main()