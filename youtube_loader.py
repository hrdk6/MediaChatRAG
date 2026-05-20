from langchain_community.document_loaders import YoutubeLoader

def load_youtube_transcript(url):

    loader=YoutubeLoader.from_youtube_url(
        url,
        add_video_info=False
    )
    docs=loader.load()
    return docs
