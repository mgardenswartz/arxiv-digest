# arXiv Digest

A customizable tool that automatically fetches, ranks, and summarizes the latest arXiv papers based on your research interests. The system runs weekly using GitHub Actions and delivers results to your email.

## Quick Setup

### 1. Fork this Repository

Click the "Fork" button at the top right of the repository page to create your own copy.

### 2. Configure Your Research Interests

Edit the `USER_PROFILE` section in `arxiv_digest.py` to match your research interests:

```python
USER_PROFILE = {
    # Primary research areas (weighted more heavily)
    'primary_interests': [
        'control systems',
        'robotics',
        'optimal control',
        # Add your own interests here
    ],

    # Secondary interests (weighted less heavily)
    'secondary_interests': [
        'reinforcement learning',
        'system identification',
        # Add your secondary interests here
    ],

    # Favorite authors (papers by these authors will be ranked higher)
    'favorite_authors': [
        'John Doe',
        'Jane Doe',
        # Add your favorite authors here
    ],

    # Positive keywords (presence increases paper relevance)
    'positive_keywords': [
        'stability',
        'robustness',
        # Add keywords that make papers more relevant to you
    ],

    # Negative keywords (presence decreases paper relevance)
    'negative_keywords': [
        'survey',
        'review',
        # Add keywords that make papers less relevant to you
    ]
}
```

### 3. Configure Repository Secrets

To securely store API keys and email credentials:

1. Go to your forked repository → Settings → Secrets and variables → Actions
2. Add the following secrets:
   - `OPENAI_API_KEY`: Your OpenAI API key (for GPT-4o Mini summaries)
   - `EMAIL_ADDRESS`: Your email address (if using email delivery)
   - `EMAIL_PASSWORD`: Your email app password
   - `RECIPIENT_EMAIL`: Where to send the digest

### 4. Adjust the Schedule (Optional)

To change when the digest runs:

1. Open `.github/workflows/arxiv_digest.yml`
2. Modify the cron expression in the schedule section:

```yaml
on:
  schedule:
    # This runs every Monday at 12:00 PM UTC
    - cron: "0 12 * * 1"
```

## Email Setup

If you're using Gmail, create an App Password:

- Go to your Google Account → Security → App passwords
- Select "Mail" and your device, then generate a new app password
- Use this password in the `EMAIL_PASSWORD` secret

For other email providers, update the `smtp_server` and `smtp_port` in the config.

## Manual Execution

To run the workflow manually:

1. Go to the "Actions" tab in your repository
2. Select the "Weekly arXiv Digest" workflow
3. Click "Run workflow" → "Run workflow"

## License

This project is licensed under the MIT License - see the LICENSE file for details.
