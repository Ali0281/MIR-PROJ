from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json

# my imports :
import requests


class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    # source : https://coderslegacy.com/beautifulsoup-user-agents/
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        # TODO √
        self.crawling_threshold = crawling_threshold
        self.not_crawled = deque()
        self.crawled = set()
        self.added_ids = set()
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        # TODO :: id_ is complete?
        URL = URL.lower()
        parsed = URL.split("/")
        id_ = parsed[parsed.index("title") + 1]
        return id_

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        # TODO √
        with open('IMDB_crawled.json', 'w') as f:
            for i in self.crawled:
                f.write(i + "\n")

        with open('IMDB_not_crawled.json', 'w') as f:
            for i in self.not_crawled:
                f.write(i + "\n")

        with open('IMDB_added_ids.json', 'w') as f:
            for i in self.added_ids:
                f.write(i + "\n")
    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        # TODO √
        with open('IMDB_crawled.json', 'r') as f:
            self.crawled = set()
            for line in f:
                self.crawled.add(line)

        with open('IMDB_not_crawled.json', 'r') as f:
            self.not_crawled = deque()
            for line in f:
                self.not_crawled.append(line)

        with open('IMDB_added_ids.json', 'r') as f:
            self.added_ids = set()
            for line in f:
                self.added_ids.add(line)
    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        # TODO √
        result = None
        try:
            response = requests.get(URL, headers = self.headers)
            if response.status_code != 200 :
                print(f"couldn't get {URL}, status : {response.status_code}")
            else:
                print(f"got {URL} successfully, state : 200")
                result = response
        except Exception as e:
            print(f"couldn't get {URL}, exception : {e}")
        finally:
            return result



    def handle_locality(self, base, url):
        if url.startswith(base) : return url
        return base + url

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        # TODO update self.not_crawled and self.added_ids √
      #  try :
        response = self.crawl(self.top_250_URL)
        if response is None: raise Exception(f"couldn't get {self.top_250_URL}")
        soup = BeautifulSoup(response.content, 'html.parser')
        list_movies = soup.find("ul", class_ = "ipc-metadata-list ipc-metadata-list--dividers-between sc-a1e81754-0 eBRbsI compact-list-view ipc-metadata-list--base")
        movies = list_movies.find_all("a", class_ = "ipc-title-link-wrapper")
        for movie in movies:
            link = self.handle_locality(base = "https://www.imdb.com",url = movie.get("href"))
            # print(link)
            self.crawl_page_info(link)
            self.not_crawled.append(link)
            self.added_ids.add(self.get_id_from_URL(link))
        # TODO : may not be needed
        # self.crawled.add(self.top_250_URL)

        """ except Exception as e:
        print(f"couldn't retrieve all top movies, exception : {e}")

        finally:
        pass"""


    def get_imdb_instance(self):
        # TODO
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
        TODO: 
            replace WHILE_LOOP_CONSTRAINTS with the proper constraints for the while loop. √
            replace NEW_URL with the new URL to crawl. √
            replace THERE_IS_NOTHING_TO_CRAWL with the condition to check if there is nothing to crawl. √
            delete help variables. √
            -mine : fix locks ,test

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """

        self.extract_top_250()
        futures = []
        crawled_counter = 0

        with ThreadPoolExecutor(max_workers=20) as executor:
            while crawled_counter < self.crawling_threshold:
                self.add_queue_lock.acquire()
                URL = self.not_crawled.popleft()
                self.add_queue_lock.release()

                futures.append(executor.submit(self.crawl_page_info, URL))
                if len(self.not_crawled) == 0:
                    wait(futures)
                    futures = []

    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.
        
        Parameters
        ----------
        URL: str
            The URL of the site
        """
        print("new iteration")
        # TODO
        response = self.crawl(URL)

        movie = dict()
        self.extract_movie_info(response, movie, URL)
        print(movie)


        if movie['related_links'] is None : return
        for link in movie['related_links']:
            if self.get_id_from_URL(link) in self.added_ids : continue
            self.added_ids.add(self.get_id_from_URL(link))
            self.not_crawled.append(link)

    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        # TODO √
        #try:
        soup = BeautifulSoup(res.content, 'html.parser')

        res_summaries = self.crawl(self.get_summary_link(URL))
        res_reviews = self.crawl(self.get_review_link(URL))
        print(res_summaries)
        print(res_reviews)
        soup_summaries = BeautifulSoup(res_summaries.content, 'html.parser')
        soup_reviews = BeautifulSoup(res_reviews.content, 'html.parser')

        movie['title'] = self.get_title(soup)
        movie['first_page_summary'] = self.get_first_page_summary(soup)
        movie['release_year'] = self.get_release_year(soup)
        movie['mpaa'] = self.get_mpaa(soup)
        movie['budget'] = self.get_budget(soup)
        movie['gross_worldwide'] = self.get_gross_worldwide(soup)
        movie['directors'] = self.get_directors(soup)
        movie['writers'] = self.get_writers(soup)
        movie['stars'] = self.get_stars(soup)
        movie['related_links'] = self.get_related_links(soup)
        movie['genres'] = self.get_genres(soup)
        movie['languages'] = self.get_languages(soup)
        movie['countries_of_origin'] = self.get_countries_of_origin(soup)
        movie['rating'] = self.get_rating(soup)
        movie['summaries'] = self.get_summaries(soup_summaries)
        movie['synopsis'] = self.get_synopsis(soup_summaries)
        movie['reviews'] = self.get_reviews_with_scores(soup_reviews)

        """except Exception as e:
            print(f"couldn't extract movie info, exception : {e}")
        finally:
            pass
"""
    def get_summary_link(self, url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            url = self.handle_locality(base = "https://www.imdb.com", url = url)
            return url.split("?")[0] + "plotsummary"
        except Exception as e:
            print(f"failed to get summary link, exception : {e}")
        finally:
            pass

    def get_review_link(self, url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            url = self.handle_locality(base="https://www.imdb.com", url=url)
            return url.split("?")[0] + "reviews"
        except Exception as e:
            print(f"failed to get review link, exception : {e}")
        finally:
            pass

    def get_title(self, soup):
        """
        Get the title of the movie from the soup


        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        # TODO √
        title = None
        try:
            title_h1 = soup.find("h1",  {"class" : "sc-d8941411-0","data-testid" : "hero__pageTitle"})
            if title_h1 is None : raise Exception("unknown title type")
            title = title_h1.span.string
        except Exception as e:
            print(f"failed to get title, exception : {e}")
        finally:
            return title

    def get_first_page_summary(self, soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        # TODO √
        summary = None
        try:
            summary_p = soup.find("p", {"class" : "sc-466bb6c-3", "data-testid" : "plot"})
            if summary_p is None:
                raise Exception("unknown summary type")
            summary = summary_p.span.string
        except Exception as e:
            print(f"failed to get first page summary, exception : {e}")
        finally:
            return summary


    def get_directors(self, soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        # TODO √
        writer_names = []
        try:
            top_credits_div = soup.find("div", class_="sc-410d722f-1")
            writers_section = top_credits_div.find_all("li", class_="ipc-metadata-list__item")[1]
            writers = writers_section.find_next("ul",
                                                class_="ipc-inline-list ipc-inline-list--show-dividers ipc-inline-list--inline ipc-metadata-list-item__list-content baseAlt").find_all(
                "li", class_="ipc-inline-list__item")
            for writer in writers:
                writer_names.append(writer.a.string)
            if len(writer_names) == 0: raise Exception("unknown writers field")
        except Exception as e:
            print(f"failed to get writers, exception : {e}")
        finally:
            return writer_names

    def get_stars(self, soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        # TODO √
        stars_names = []
        try:
            top_credits_div = soup.find("div", class_="sc-410d722f-1")
            stars_section = top_credits_div.find_all("li", class_="ipc-metadata-list__item")[2]
            stars = stars_section.find_next("ul",
                                                class_="ipc-inline-list ipc-inline-list--show-dividers ipc-inline-list--inline ipc-metadata-list-item__list-content baseAlt").find_all(
                "li", class_="ipc-inline-list__item")
            for star in stars:
                stars_names.append(star.a.string)
            if len(stars_names) == 0: raise Exception("unknown writers field")
        except Exception as e:
            print(f"failed to get writers, exception : {e}")
        finally:
            return stars_names

    def get_writers(self, soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        # TODO √
        writer_names = []
        try:
            top_credits_div = soup.find("div", class_="sc-410d722f-1")
            writers_section = top_credits_div.find_all("li", class_="ipc-metadata-list__item")[1]
            writers = writers_section.find_next("ul",
                                                    class_="ipc-inline-list ipc-inline-list--show-dividers ipc-inline-list--inline ipc-metadata-list-item__list-content baseAlt").find_all(
                "li", class_="ipc-inline-list__item")
            for writer in writers:
                writer_names.append(writer.a.string)
            if len(writer_names) == 0 : raise Exception("unknown writers field")
        except Exception as e:
            print(f"failed to get writers, exception : {e}")
        finally:
            return writer_names

    def get_related_links(self, soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        # TODO √
        related_links = []
        try:
            posters = soup.find_all("a", class_="ipc-poster-card__title ipc-poster-card__title--clamp-2 ipc-poster-card__title--clickable")
            for poster in posters:
                related_links.append(self.handle_locality(base = "https://www.imdb.com",url =poster.get("href")))
            if len(related_links) == 0: raise Exception("unknown related field")
        except Exception as e:
            print(f"failed to get related links, exception : {e}")
        finally:
            return related_links

    def get_summaries(self, soup):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        # TODO √
        summaries = []
        try:
            li_summaries = soup.find_all("li", {"class" : "ipc-metadata-list__item", "data-testid" : "list-item"})[:-1]
            for summary in li_summaries:
                summaries.append(summary.text)
            if len(summaries) == 0: raise Exception("unknown related field")
        except Exception as e:
            print(f"failed to get summaries, exception : {e}")
        finally:
            return summaries

    def get_synopsis(self, soup):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        # TODO √
        synopsis = None
        try:
            synopsis = soup.find_all("li", {"class" : "ipc-metadata-list__item", "data-testid" : "list-item"})[-1].text
            if synopsis is None: raise Exception("unknown related field")
        except Exception as e:
            print(f"failed to get synopsis, exception : {e}")
        finally:
            return synopsis


    def get_reviews_with_scores(self, soup):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get reviews")

    def get_genres(self, soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        # TODO √
        pass
        """genres = []
        try:
            group_section = soup.find("ul", class_="ipc-metadata-list ipc-metadata-list--dividers-all sc-ab73d3b3-1 jzyXiW ipc-metadata-list--base")
            print(group_section.prettify())
            # genres_section = group_section.find_all("li", {"class" : "ipc-metadata-list__item", "data-testid" : "storyline-taglines"})[1]
            # genres_a = genres_section.find_all("a", class_ = "ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link")
            # for genre in genres_a:
                # print(genres_a.string)
                # genres.append(genres_a.string)
            # if len(genres) == 0: raise Exception("unknown genres field")
        except Exception as e:
            print(f"failed to get genres, exception : {e}")
        finally:
            return genres"""

    def get_rating(self, soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get rating")

    def get_mpaa(self, soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get mpaa")

    def get_release_year(self, soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get release year")

    def get_languages(self, soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get languages")
            return None

    def get_countries_of_origin(self, soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get countries of origin")

    def get_budget(self, soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        # TODO √
        budget_amounts = []
        try:
            box_office = soup.find("div", {"class" : "sc-f65f65be-0","data-testid" : "title-boxoffice-section"})
            gross_section = box_office.find("li", {"class" : "ipc-metadata-list__item","data-testid" : "title-boxoffice-budget"})
            for gross in gross_section.div.find_all("li"):
                budget_amounts.append(gross.span.text)
                print(gross.span.text)
            if len(budget_amounts) == 0: raise Exception("unknown writers field")
        except Exception as e:
            print(f"failed to get gross worldwide, exception : {e}")
        finally:
            return budget_amounts


    def get_gross_worldwide(self, soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        # TODO √
        gross_amount = None
        try:
            box_office = soup.find("div", {"class" : "sc-f65f65be-0","data-testid" : "title-boxoffice-section"})
            gross_section = box_office.find("li", {"class" : "ipc-metadata-list__item","data-testid" : "title-boxoffice-cumulativeworldwidegross"})
            for gross in gross_section.div.find("li").span.text:
                gross_amounts.append(gross.span.text)
            if len(gross_amounts) == 0: raise Exception("unknown writers field")
        except Exception as e:
            print(f"failed to get gross worldwide, exception : {e}")
        finally:
            return gross_amounts



def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=600)
    imdb_crawler.extract_top_250()

    # imdb_crawler.read_from_file_as_json()
    """
    imdb_crawler.start_crawling()
    imdb_crawler.write_to_file_as_json()
"""

if __name__ == '__main__':
    main()
