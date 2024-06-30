import json
from typing import Dict, List

from bs4 import BeautifulSoup
import requests

from Logic.core.utility import IMDbCrawler
from .core.search import SearchEngine
from .core.utility.spell_correction import SpellCorrection
from .core.utility.snippet import Snippet
from .core.indexer.indexes_enum import Indexes, Index_types
from typing import Dict, List

from Logic.core.indexer.indexes_enum import Indexes
from Logic.core.utility.preprocess import Preprocessor
from Logic.core.utility.spell_correction import (SpellCorrection)
from Logic.core.search import SearchEngine

movies_dataset = None  # TODO
# with open("C:/Users/Ali/PycharmProjects/MIR-PROJ/Logic/core/preprocess.json", "r") as f:
#    movies_dataset = json.load(f)
with open("C:/Users/Ali/PycharmProjects/MIR-PROJ/Logic/core/IMDB_movies.json", "r") as f:
    movies_dataset = json.load(f)

search_engine = SearchEngine()


def correct_text(text: str, all_documents: List[str]) -> str:
    """
    Correct the give query text, if it is misspelled using Jacard similarity

    Paramters
    ---------
    text: str
        The query text
    all_documents : list of str
        The input documents.

    Returns
    str
        The corrected form of the given text
    """
    # TODO: You can add any preprocessing steps here, if needed!
    # TODO : note : seems there is nothing to do in this file for this phase but this section! there is no need for testing, just check out spell_correction.py
    pre = Preprocessor([{}], "C:/Users/Ali/PycharmProjects/MIR-PROJ/Logic/core/stopwords.txt")
    pre.preprocess()
    data = pre.documents
    spell_correction_obj = SpellCorrection(data)
    return spell_correction_obj.spell_check(text)


def search(
        query: str,
        max_result_count: int,
        method: str = "ltn-lnn",
        weights: list = [0.3, 0.3, 0.4],
        should_print=False,
        preferred_genre: str = None,
        unigram_smoothing=None,
        alpha=0.5,
        lamda=0.5,
):
    """
    Finds relevant documents to query

    Parameters
    ---------------------------------------------------------------------------------------------------
    max_result_count: Return top 'max_result_count' docs which have the highest scores.
                      notice that if max_result_count = -1, then you have to return all docs

    mode: 'detailed' for searching in title and text separately.
          'overall' for all words, and weighted by where the word appears on.

    where: when mode ='detailed', when we want search query
            in title or text not both of them at the same time.

    method: 'ltn.lnn' or 'ltc.lnc' or 'OkapiBM25'

    preferred_genre: A list containing preference rates for each genre. If None, the preference rates are equal.

    Returns
    ----------------------------------------------------------------------------------------------------
    list
    Retrieved documents with snippet
    """
    weights = {
        Indexes.STARS: weights[0],
        Indexes.GENRES: weights[1],
        Indexes.SUMMARIES: weights[2]
    }
    """weights = {
        Indexes.STARS: 0.1,
        Indexes.GENRES: 0.1,
        Indexes.SUMMARIES: 0.8
    }"""

    return search_engine.search(
        query, method, weights, max_results=max_result_count, safe_ranking=True, smoothing_method=unigram_smoothing,
        alpha=alpha, lamda=lamda
    )


def get_movie_by_id(id: str, movies_dataset: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Get movie by its id

    Parameters
    ---------------------------------------------------------------------------------------------------
    id: str
        The id of the movie

    movies_dataset: List[Dict[str, str]]
        The dataset of movies

    Returns
    ----------------------------------------------------------------------------------------------------
    dict
        The movie with the given id
    """

    """result = movies_dataset.get(
        id,
        {
            "Title": "This is movie's title",
            "Summary": "This is a summary",
            "URL": "https://www.imdb.com/title/tt0111161/",
            "Cast": ["Morgan Freeman", "Tim Robbins"],
            "Genres": ["Drama", "Crime"],
            "Image_URL": "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg",
        },
    )"""
    result = None
    for document in movies_dataset:
        if document["id"] == id:
            result = document
            break
    if result is None:
        result = movies_dataset[0]
    ##########################################################################################
    crawler = IMDbCrawler()
    respond = crawler.crawl("https://www.imdb.com/title/" + result["id"])
    soup = BeautifulSoup(respond.content, 'html.parser')
    img_url = ""
    try:
        img_ = soup.find("img", {"class": "ipc-image"})
        if img_ is None: raise Exception("unknown image field")
        img_url = img_.get("src")
    except Exception as e:
        print(f"failed to get img_url, exception : {e}")
    finally:
        img_url = img_url if img_url is not None else ""

    '''
    video_url = ""
    try:
        video_ = soup.find("div", {"class": "jw-wrapper jw-reset"})
        print(video_)
        video_ = video_.find("video")
        print(video_)
        if video_ is None: raise Exception("unknown video field")
        video_url = video_.get("src")
        print(video_, end="\n\n\n\n")
        """
        scripts = soup.find_all('script', type='application/json')
        for script in scripts:
            try:
                json_data = json.loads(script.string)
                if 'video' in json_data:
                    video_url = json_data['video']['playbackUrls'][0]
                    break
            except (json.JSONDecodeError, KeyError):
                continue"""

        if not video_url:
            raise Exception("unknown video field")

        print("Video URL:", video_url)
    except Exception as e:
        print(f"failed to get video_url, exception : {e}")
    finally:
        video_url = video_url if video_url is not None else ""
        
    '''

    url = f"https://api.themoviedb.org/3/movie/{result["id"]}/videos?language=en-US"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIzNThmZDhmNjE0YWFmNmIyNzJiZmVhY2Q5MDE1NGU3OSIsIm5iZiI6MTcxOTc1NTM0Mi42OTExNiwic3ViIjoiNjY4MTVmOWVmMjkwZmZkYzIwMTYzOTFiIiwic2NvcGVzIjpbImFwaV9yZWFkIl0sInZlcnNpb24iOjF9._xpi4fvuzHK6vc_sm1nIiNLkF4kzLNIqsSfzpo9dyCU"
    }
    videos = []
    names = []
    video_url = ""
    name_vid = ""
    response = requests.get(url, headers=headers)
    flag = True
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])

        for video in results:
            name = video.get('name')
            key = video.get('key')
            if name and key:
                youtube_url = f"https://www.youtube.com/watch?v={key}"
                if flag:
                    video_url = youtube_url
                    name_vid = name
                    flag = False
                videos.append(youtube_url)
                names.append(name)
    else:
        print(f"Failed to fetch data: {response.status_code} - {response.text}")
    #############################################################################################
    if img_url == "":
        result["Image_URL"] = (
            "https://m.media-amazon.com/images/M/MV5BNDE3ODcxYzMtY2YzZC00NmNlLWJiNDMtZDViZWM2MzIxZDYwXkEyXkFqcGdeQXVyNjAwNDUxODI@._V1_.jpg"
        )
    else:
        result["Image_URL"] = (
            img_url
        )

    if video_url == "":
        result["Video_URL"] = (
            "https://imdb-video.media-imdb.com/vi3877612057/1434659607842-pgv4ql-1616202333253.mp4?Expires=1719838038&Signature=s0FFUmNnn0NoZaARVOj7jDOMVdI7yDAdgMa5~xOEwVFO9qcWriOuCjibJZkip0YYZwsmwBU3BRPnK9cqsoXvnfoeVNkGVSWRJBDMgUmGlRaPcmG9Ddfls3krrfRRUpwFe435Q4C9phyyVtTOmmsZAm2WEd5rbj7f4LHRP-S7F8HaARmUnEYnCozTXFGbBLmT41VfuFbeIBNPW6anu2jsKb0K~XC2TETj9Jrf3bJc0p9WDcTxRQ~C~xmKyMMeK70kShxegpOI2fxuyxPaSjwq~Qq4F8K2lgzqOWHi7vdR1tB3pd~m5smxaIR91l7GzpE5mL-ey1TAUNUJYhcTrt0Ivw__&Key-Pair-Id=APKAIFLZBVQZ24NQH3KA"
        )
        result["Video_name"] = ("coudnt find; but watch this cool vid!")
    else:
        result["Video_URL"] = (
            video_url
        )
        result["Video_name"] = (name_vid)

    result["videos"] = videos
    result["names"] = names
    result["URL"] = (
        f"https://www.imdb.com/title/{result.get('id', 'NOT FOUND')}"  # The url pattern of IMDb movies
    )
    return result
