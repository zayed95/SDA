import re, time
import requests
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

url = "https://mastodon.social/api/v1/timelines/tag/trump"
limit = 1000
max_id = None
posts = []

while len(posts) < limit:

    params = {'limit': 50}
    if max_id:
        params['max_id'] = max_id

    r = requests.get(
        url=url,
        params=params
    )

    if r.status_code == 200:
        data = r.json()

        logger.info("200 OK")

        if not data:
            break

        for post in data:
            if post.get('language') == "en" :
                content = re.sub('<[^<]+?>', '', post['content'])
                if len(content) > 30:

                    posts.append({
                        "created_at": post['created_at'],
                        "username": post['account']['username'],
                        "text": content.replace("\n", " ").strip()
                    })
        max_id = data[-1]['id']
        time.sleep(0.3)
    
    else:
        logger.error(f"Error while requesting: {r.status_code}")
        break


df = pd.DataFrame(posts)

file_name = "data/scraped-data.csv"
df.to_csv(file_name, index=False, encoding='utf-8')
