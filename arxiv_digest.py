import datetime
import os
import random
import re
import smtplib
import ssl
import time
import urllib.parse
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import feedparser
import markdown
import nltk

try:
    # Try to tokenize a test sentence
    from nltk.tokenize import word_tokenize

    word_tokenize("Test sentence")
except LookupError:
    # If resources are missing, download them
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    nltk.download("wordnet")

import openai
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI API configuration
OPENAI_CONFIG = {
    "api_key": os.environ.get("OPENAI_API_KEY"),
    "model": "gpt-4o-mini",
    "use_ai_summaries": False,
}

# Email configuration
EMAIL_CONFIG = {
    "send_email": True,
    "sender_email": os.environ.get("EMAIL_ADDRESS"),
    "sender_password": os.environ.get("EMAIL_PASSWORD"),
    "recipient_email": os.environ.get("RECIPIENT_EMAIL"),
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 465,
}

# User Preferences - Edit these to match your interests
USER_PROFILE = {
    # Primary research areas (weighted more heavily)
    "primary_interests": [
        "adaptive control",
        "graph neural networks",
        "nonlinear systems",
        "reinforcement learning",
    ],
    # Secondary interests (weighted less heavily)
    "secondary_interests": [
        "multi-agent systems",
        "autonomous vehicles",
        "quadcopter control",
    ],
    # Favorite authors (papers by these authors will be ranked higher)
    "favorite_authors": [
        "Fernando Gama",
        "Alejandro Ribeiro",
        "Amanda Prorok",
        "Hans Riess",
    ],
    # Positive keywords (presence increases paper relevance)
    "positive_keywords": [
        "stability",
        "Lyapunov",
        "nonlinear",
        "deep learning",
    ],
    # Negative keywords (presence decreases paper relevance)
    "negative_keywords": ["survey", "review", "introductory"],
}

# Configure these variables based on user profile
SEARCH_QUERIES = USER_PROFILE["primary_interests"]
MAX_RESULTS = 30  # Number of papers to fetch per query
TOP_PAPERS = 7  # Number of papers to include in the digest
OUTPUT_FILE = "arxiv_digest.md"  # Output file name

# Define emojis for paper categories
PAPER_EMOJIS = {
    "control systems": "🎮",
    "robotics": "🤖",
    "optimal control": "⚙️",
    "model predictive control": "📊",
    "reinforcement learning": "🧠",
    "system identification": "🔍",
    "autonomous vehicles": "🚗",
    "drone": "✈️",
    "stability": "⚖️",
    "robustness": "🛡️",
    "nonlinear": "📈",
    "embedded": "💻",
    "default": "📝",  # Default emoji if no match
}


def get_emoji_for_paper(paper):
    """Select an appropriate emoji for a paper based on its content."""
    title_and_summary = (paper["title"] + " " + paper["summary"]).lower()

    for keyword, emoji in PAPER_EMOJIS.items():
        if keyword.lower() in title_and_summary:
            return emoji

    # If no match, use default or a random academic emoji
    academic_emojis = ["📚", "🔬", "🧪", "📊", "📈", "🔭", "🧮"]
    return random.choice(academic_emojis)


def clean_text(text):
    """Clean and preprocess text."""
    # Remove special characters and digits
    text = re.sub(r"\W+|\d+", " ", text.lower())

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


def fetch_arxiv_papers():
    """Fetch papers from arXiv based on search queries."""
    all_papers = []

    # Combine primary and secondary interests for searching
    all_queries = (
        USER_PROFILE["primary_interests"] + USER_PROFILE["secondary_interests"]
    )

    for query in all_queries:
        # Format the query for the URL - properly encode it
        formatted_query = urllib.parse.quote(query)

        # Get the current date minus 7 days
        week_ago = datetime.datetime.now() - datetime.timedelta(days=7)
        date_str = week_ago.strftime("%Y%m%d")

        # Create the arXiv API URL
        url = f"http://export.arxiv.org/api/query?search_query=all:{formatted_query}&start=0&max_results={MAX_RESULTS}&sortBy=submittedDate&sortOrder=descending"

        try:
            # Parse the RSS feed
            print(f"Fetching papers for query: {query}")
            feed = feedparser.parse(url)

            # Extract relevant information
            for entry in feed.entries:
                # Check if this paper was already added (to avoid duplicates)
                if not any(paper["link"] == entry.link for paper in all_papers):
                    paper = {
                        "title": entry.title,
                        "authors": ", ".join(author.name for author in entry.authors),
                        "summary": entry.summary,
                        "link": entry.link,
                        "published": entry.published,
                        "query": query,
                    }
                    all_papers.append(paper)

        except Exception as e:
            print(f"Error fetching papers for query '{query}': {str(e)}")

    print(f"Total papers fetched: {len(all_papers)}")
    return all_papers


def rank_papers(papers, user_profile=USER_PROFILE):
    """Rank papers based on relevance to user interests."""
    if not papers:
        return []

    # Combine user interests with different weights
    # Primary interests get mentioned twice for higher weight
    user_interests = " ".join(
        user_profile["primary_interests"]
        + user_profile["primary_interests"]
        + user_profile["secondary_interests"]
    )

    # Clean user interests for TF-IDF
    clean_interests = clean_text(user_interests)

    # Clean paper abstracts
    abstracts = [clean_text(paper["summary"]) for paper in papers]

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Add user interests to the documents to be vectorized
    documents = abstracts + [clean_interests]

    # Transform documents to TF-IDF features
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Calculate similarity between each paper and user interests
    similarities = cosine_similarity(
        tfidf_matrix[:-1],  # All papers
        tfidf_matrix[-1].reshape(1, -1),  # User interests
    ).flatten()

    # Add similarity score to papers
    for i, paper in enumerate(papers):
        paper["relevance_score"] = similarities[i]

        # Adjust scores based on authors
        author_bonus_applied = False
        for author in user_profile["favorite_authors"]:
            if author.lower() in paper["authors"].lower():
                paper["relevance_score"] += 0.2
                author_bonus_applied = True
                break  # Only apply the bonus once

        # Adjust scores based on positive keywords
        keyword_bonus_applied = False
        for keyword in user_profile["positive_keywords"]:
            if (
                keyword.lower() in paper["title"].lower()
                or keyword.lower() in paper["summary"].lower()
            ):
                if not keyword_bonus_applied:
                    paper["relevance_score"] += 0.1
                    keyword_bonus_applied = True
                    # We continue checking to potentially apply multiple bonuses

        # Adjust scores based on negative keywords
        for keyword in user_profile["negative_keywords"]:
            if (
                keyword.lower() in paper["title"].lower()
                or keyword.lower() in paper["summary"].lower()
            ):
                paper["relevance_score"] -= 0.1

    # Sort papers by relevance score
    ranked_papers = sorted(papers, key=lambda x: x["relevance_score"], reverse=True)

    return ranked_papers


def summarize_with_gpt(abstract, title, max_words=50):
    """Summarize an abstract using GPT-4o Mini."""
    if not OPENAI_CONFIG["use_ai_summaries"]:
        # Fall back to simple summarization if AI summaries are disabled
        return summarize_abstract(abstract, max_words)

    try:
        # Configure OpenAI client
        client = openai.OpenAI(api_key=OPENAI_CONFIG["api_key"])

        # Create a prompt for summarization
        prompt = f"""Summarize the following research paper abstract in a concise, informative way. 
        Focus on the key innovation, methodology, and results. Be specific about technical details.
        Keep the summary under 50 words.
        
        Title: {title}
        Abstract: {abstract}
        """

        # Call the OpenAI API
        response = client.chat.completions.create(
            model=OPENAI_CONFIG["model"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledgeable AI assistant specializing in summarizing scientific research papers. Your summaries are concise, technically precise, and highlight the most important aspects of the research.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            temperature=0.3,  # Lower temperature for more focused, deterministic responses
        )

        # Extract the summary from the response
        ai_summary = response.choices[0].message.content.strip()

        print(f"Generated AI summary for: {title[:30]}...")
        return ai_summary

    except Exception as e:
        print(f"Error generating AI summary: {str(e)}")
        print("Falling back to simple summarization...")
        return summarize_abstract(abstract, max_words)


def summarize_abstract(abstract, max_words=50):
    """Create a shorter summary of the abstract."""
    # For a more sophisticated summary, you could use a library like
    # transformers with a pre-trained model, but here's a simple version:
    sentences = re.split(r"(?<=[.!?])\s+", abstract)
    summary = " ".join(sentences[:2])  # Just take the first two sentences

    # Truncate to max_words
    words = summary.split()
    if len(words) > max_words:
        summary = " ".join(words[:max_words]) + "..."

    return summary


def process_top_papers_with_ai_summaries(papers):
    """Process the top papers to add AI-generated summaries."""
    if not OPENAI_CONFIG["use_ai_summaries"] or not papers:
        return papers

    print("Generating AI summaries for top papers...")

    for i, paper in enumerate(papers):
        print(f"Processing paper {i + 1}/{len(papers)}")
        paper["ai_summary"] = summarize_with_gpt(paper["summary"], paper["title"])
        # Add a small delay to avoid hitting API rate limits
        time.sleep(1)

    return papers


def create_markdown_content(papers):
    """Create markdown content from papers."""
    if not papers:
        return "# arXiv Digest\n\nNo relevant papers found this week."

    today = datetime.datetime.now().strftime("%Y-%m-%d")

    markdown = f"# arXiv Digest - {today}\n\n"

    # Add a bulleted list with titles and emojis at the top
    markdown += "## Quick Links\n\n"
    for i, paper in enumerate(papers):
        # Clean title and get appropriate emoji
        clean_title = paper["title"].replace("\n", " ").replace("  ", " ")
        emoji = get_emoji_for_paper(paper)

        # Add the bulleted item with emoji linking directly to arXiv paper
        markdown += f"* {emoji} [{clean_title}]({paper['link']})\n"

    markdown += "\n---\n\n"
    markdown += f"Top {len(papers)} relevant papers from arXiv this week.\n\n"

    for i, paper in enumerate(papers):
        # Fix for multi-line titles - replace newlines with spaces
        clean_title = paper["title"].replace("\n", " ").replace("  ", " ")
        emoji = get_emoji_for_paper(paper)

        # Make the title a link to the arXiv paper
        markdown += f"## {i + 1}. {emoji} [{clean_title}]({paper['link']})\n\n"

        # Add relevant keywords that increased the score
        relevant_keywords = []
        for keyword in USER_PROFILE["positive_keywords"]:
            if (
                keyword.lower() in paper["title"].lower()
                or keyword.lower() in paper["summary"].lower()
            ):
                relevant_keywords.append(keyword)

        # Check for favorite authors
        favorite_authors_present = []
        for author in USER_PROFILE["favorite_authors"]:
            if author.lower() in paper["authors"].lower():
                favorite_authors_present.append(author)

        markdown += f"**Authors:** {paper['authors']}"
        if favorite_authors_present:
            markdown += f" (including favorite author(s): {', '.join(favorite_authors_present)})"
        markdown += "\n\n"

        markdown += f"**Published:** {paper['published']}\n\n"

        # Use AI summary if available, otherwise fall back to simple summary
        if OPENAI_CONFIG["use_ai_summaries"] and "ai_summary" in paper:
            markdown += f"**Summary:** {paper['ai_summary']}\n\n"
        else:
            markdown += f"**Summary:** {summarize_abstract(paper['summary'])}\n\n"

        if relevant_keywords:
            markdown += f"**Relevant keywords:** {', '.join(relevant_keywords)}\n\n"

        # Remove the extra link line since the title is now a link
        markdown += f"**Relevance Score:** {paper['relevance_score']:.4f}\n\n"
        markdown += "---\n\n"

    # Add explanation of how papers were ranked
    markdown += "## How Papers Were Ranked\n\n"
    markdown += "Papers were ranked based on your preferences:\n\n"
    markdown += (
        f"**Primary Interests:** {', '.join(USER_PROFILE['primary_interests'])}\n\n"
    )
    markdown += (
        f"**Secondary Interests:** {', '.join(USER_PROFILE['secondary_interests'])}\n\n"
    )
    markdown += (
        f"**Favorite Authors:** {', '.join(USER_PROFILE['favorite_authors'])}\n\n"
    )
    markdown += (
        f"**Positive Keywords:** {', '.join(USER_PROFILE['positive_keywords'])}\n\n"
    )

    # Add note about summaries
    if OPENAI_CONFIG["use_ai_summaries"]:
        markdown += "\n**Note:** Paper summaries were generated using GPT-4o Mini to highlight key innovations and findings.\n\n"

    markdown += f"Generated on {today}\n"

    return markdown


def save_markdown_file(content, filename=OUTPUT_FILE):
    """Save content to markdown file."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Digest saved to {os.path.abspath(filename)}")


def send_email_digest(markdown_content):
    """Send the digest as an HTML email."""
    if not EMAIL_CONFIG["send_email"]:
        print(
            "Email sending is disabled. Set EMAIL_CONFIG['send_email'] to True to enable."
        )
        return False

    try:
        # Create message
        message = MIMEMultipart("alternative")
        message["Subject"] = (
            f"arXiv Digest - {datetime.datetime.now().strftime('%Y-%m-%d')}"
        )
        message["From"] = EMAIL_CONFIG["sender_email"]
        message["To"] = EMAIL_CONFIG["recipient_email"]

        # Convert markdown to HTML
        html_content = markdown.markdown(markdown_content)

        # Add some basic styling
        styled_html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
                h2 {{ color: #3498db; margin-top: 30px; }}
                a {{ color: #2980b9; text-decoration: none; }}
                a:hover {{ text-decoration: underline; }}
                hr {{ border: 0; border-top: 1px solid #eee; margin: 30px 0; }}
                .relevance {{ color: #7f8c8d; font-size: 0.9em; }}
                .authors {{ font-style: italic; }}
                .favorite-author {{ color: #27ae60; font-weight: bold; }}
                .keywords {{ color: #e74c3c; }}
                ul {{ list-style-type: none; padding-left: 0; }}
                li {{ margin-bottom: 8px; }}
                li a {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        # Attach HTML content
        message.attach(MIMEText(styled_html, "html"))

        # Create secure SSL context
        context = ssl.create_default_context()

        # Send email
        with smtplib.SMTP_SSL(
            EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"], context=context
        ) as server:
            server.login(EMAIL_CONFIG["sender_email"], EMAIL_CONFIG["sender_password"])
            server.sendmail(
                EMAIL_CONFIG["sender_email"],
                EMAIL_CONFIG["recipient_email"],
                message.as_string(),
            )

        print(f"Email digest sent to {EMAIL_CONFIG['recipient_email']}")
        return True

    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False


def generate_arxiv_digest(send_email=True):
    """Main function to generate the arXiv digest."""
    print("Fetching papers...")
    papers = fetch_arxiv_papers()

    print(f"Found {len(papers)} papers. Ranking...")
    ranked_papers = rank_papers(papers)

    print("Preparing digest with top papers...")
    top_papers = ranked_papers[:TOP_PAPERS]

    # Process papers with GPT-4o Mini if enabled
    if OPENAI_CONFIG["use_ai_summaries"]:
        top_papers = process_top_papers_with_ai_summaries(top_papers)

    markdown_content = create_markdown_content(top_papers)
    save_markdown_file(markdown_content)

    if send_email and EMAIL_CONFIG["send_email"]:
        print("Sending email digest...")
        send_email_digest(markdown_content)

    return markdown_content


# Run the digest
if __name__ == "__main__":
    print("Generating arXiv digest...")
    generate_arxiv_digest(send_email=EMAIL_CONFIG["send_email"])
    print("Done!")
