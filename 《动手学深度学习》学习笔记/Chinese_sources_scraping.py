# -*- coding: utf-8 -*-
# -*- created_time: 2019/6/21 -*-

import requests
from bs4 import BeautifulSoup

# 利用urllib.parse库中的quote函数将url中的中文转换为utf-8编码
from urllib.parse import quote
import re
import pandas as pd

import warnings
warnings.filterwarnings("ignore") # Ignore all warning messages

class Xinhua_scraping():

    def __init__(self,term=["中美", "贸易战"], pages=10, main_source = "Xinhua News"):
        self.term = term
        self.pages = pages
        self.metadata = {"language":"Chinese","main source":main_source,"source rank":1,"source country":"China"}

        # 为一个字典列表,存储所有爬取的内容
        self.contents = []
        # 为了防止爬取的内容有重复,将所有爬取的news文章的title都加入self.title中,之后每爬取一篇news文章就用其title与self.title
        # 中的所有标题进行对比,看此篇文章是否重复
        self.titles = []

    def MainPage_scraping(self):
        # 用正则表达式匹配关键词变量term是只包含一个关键词的str变量还是包含多个关键词的list变量
        term_type = re.search(r"(?<=<class ')\w{1,}(?='>)", str(type(self.term))).group()
        # print(term_type)

        if term_type == "list":
            search_term = " ".join(self.term)
            # 利用urllib.parse库中的quote函数将url中的中文转换为utf-8编码
            search_term_utf8 = quote(search_term)
            # return search_term_utf8

            # 直接在requests.get()方法中包含中文也可,因为get()方法会自动将url中的中文转换为utf-8编码
            # 在此处直接用列表构造式创建所有页数的url
            urls = ["http://so.news.cn/getNews?keyword=" + search_term_utf8 + "&curPage=" + str(page_num) +
                    "&sortField=0&searchFields=1&lang=cn" for page_num in range(1, self.pages + 1)]

        else:
            # 利用urllib.parse库中的quote函数将url中的中文转换为utf-8编码
            search_term_utf8 = quote(self.term)
            # return search_term_utf8

            # 直接在requests.get()方法中包含中文也可,因为get()方法会自动将url中的中文转换为utf-8编码
            # 在此处直接用列表构造式创建所有页数的url
            urls = ["http://so.news.cn/getNews?keyword=" + search_term_utf8 + "&curPage=" + str(page_num) +
                    "&sortField=0&searchFields=1&lang=cn" for page_num in range(1, self.pages + 1)]


        # 逐页爬取(scraping page by page)
        for url in urls:
            try:
                # 爬取返回的新华社关键词news搜索结果为一个json格式的数据(相当于api),因此直接将爬取结果用res.json()方法转换为json格式数据
                res = requests.get(url)
                res.encoding = "utf=8"
                res = res.json()

                # newsList = re.sub(r"(\r\n|\n|\r)","",res["content"]["results"][0]["des"])
                newsList = res["content"]["results"]
                for news in newsList:
                    try:
                        # 利用正则表达式模块中的re.sub()函数删除爬起的文本中的无用符号等
                        title = re.sub(r"(<font color=red>|</font>|&nbsp)", "", news["title"])

                        # 为了防止爬取的内容有重复,将所有爬取的news文章的title都加入self.title中,之后每爬取一篇news文章就用其title与self.title
                        # 中的所有标题进行对比,看此篇文章是否重复
                        if title in self.titles:
                            continue
                        else:
                            self.titles.append(title)
                            extract = news["des"]
                            keyword = news["keyword"]
                            pubtime = news["pubtime"]
                            article_url = news["url"]
                            # 通过news article的sub url,爬取文章的内容
                            article_content = self.sub_url_scraping(article_url)

                            newsDict = {"title":title,"extract":extract,"keyword":keyword ,"pubtime":pubtime,"article url":article_url,
                                        "article content":article_content}

                            # 将metadata添加进每篇article的字典中,为每篇article添加更完整的信息
                            newsDict.update(self.metadata)
                            self.contents.append(newsDict)
                            # print(title)

                    except Exception:
                        continue

            except Exception:
                continue


    def sub_url_scraping(self,sub_page_url):
        texts = "" # 必须将texts初始化为一个空str,否则在下方无法进行字符串的拼接
        try:
            # 根据文章atticle的sub url爬取文章主要内容
            res = requests.get(sub_page_url)
            res.encoding = "utf=8"
            soup = BeautifulSoup(res.content,"html.parser")

            # 文章的所有主要内容都存储在网页css结构中的p标签的文本中
            paragraphs = soup.find_all("p")
            for p in paragraphs:
                try:
                    # 删除文本段落中的多空格,换行符与空白符
                    texts = texts + re.sub(r"(\s{2,}|\n|\r)","",p.text)

                # If an errot occurs in this "p" tag, then ignore it and continue loop
                except Exception:
                    continue

            return texts

        except Exception:
            return None

    # Save scraped articles' contents as csv file
    def save_as_csv(self):
        df = pd.DataFrame(self.contents,columns=["title","extract","keyword","pubtime","article url","article content",
                                                 "language","main source","source rank","source country"])
        file_name = self.metadata["main source"] + ".csv"
        df.to_csv(file_name, index=False)


class Sina_scraping():

    def __init__(self,term=["中美", "贸易战"], pages=10, main_source = "Sina News"):
        self.term = term
        self.pages = pages
        self.metadata = {"language":"Chinese","main source":main_source,"source rank":1,"source country":"China"}

        # Store all scraped content into a dictionary list
        self.contents = []
        # In order to prevent the scraping content from being duplicated, add the title of all scraped news articles to
        # self.title, and then compare each title in the self.title with each title scraped from a news article,
        # to check whether the article is repeated
        self.titles = []

    def MainPage_scraping(self):
        # Matching a keyword variable term with a regular expression is a str variable containing only one keyword
        # or a list variable containing multiple keywords
        term_type = re.search(r"(?<=<class ')\w{1,}(?='>)", str(type(self.term))).group()
        # print(term_type)

        if term_type == "list":
            search_term = " ".join(self.term)
            # Convert the Chinese in the url to utf-8 encoding using the quote function in the urllib.parse library
            search_term_utf8 = quote(search_term)
            # return search_term_utf8

        else:
            # Convert the Chinese in the url to utf-8 encoding using the quote function in the urllib.parse library
            search_term_utf8 = quote(self.term)
            # return search_term_utf8

        # It is also possible to include Chinese directly in the requests.get() method, because the get() method
        # will automatically convert the Chinese in the url to utf-8 encoding.
        # Create a url of all pages directly using the list constructor here
        urls = ["https://search.sina.com.cn/?c=news&ie=utf-8&q=" + search_term_utf8 + "&col=&range=&source=&from=&country=&size=&time=&a=&page="
            + str(page_num) + "&pf=0&ps=0&dpc=1" for page_num in range(1, self.pages + 1)]

        # Using the information of headers of browser to avoid the anti-scraping measure in website
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko'
                                 ') Chrome/65.0.3325.146 Safari/537.36',
                   'Cookie': 'd_c0="AADC7hDPuAuPTl47f3CKGWCcdFaEiAlqUJU=|1494160873"; _zap=948b68c0-7b99-4f88-8d'
                             '86-60bbdfe1d6c0; q_c1=65ef097320d64c8bb8617943d52dd592|1507048007000|1494160866000'
                             '; z_c0="2|1:0|10:1510560275|4:z_c0|92:Mi4xZXFrRkFnQUFBQUFBQU1MdUVNLTRDeVlBQUFCZ0FsV'
                             'k5FNkQyV2dBVVZRSW5pbUlzMHh3VlFCZjd2ak4zM3dRM09n|c0dcda66cfc727d786453bdc3d0a1fc00e3'
                             'cbeb7f34b723755652ed3b07c0016"; __utma=51854390.1508934931.1494160889.1510566336.15'
                             '10566336.5; __utmz=51854390.1510566336.4.3.utmcsr=zhihu.com|utmccn=(referral)|utmcm'
                             'd=referral|utmcct=/question/46417790; __utmv=51854390.100--|2=registration_date=201'
                             '50826=1^3=entry_date=20150826=1; q_c1=65ef097320d64c8bb8617943d52dd592|152083638100'
                             '0|1494160866000; __DAYU_PP=2rVjnUEImjImJIABArJeffffffff858bab885106; aliyungf_tc=AQ'
                             'AAALRrYxtqPgoAPde0AfHsaJjuBFa0; _xsrf=c86800b3-2052-4bf6-a49b-94e513acaa42'}


        # Based on all urls, scraping page by page
        for url in urls:
            try:
                # Based on url, scrape the information from the source.
                # Note, the information of headers of browser needs to be inputted into the requests.get() function,
                # to avoid the anti-scraping measure in website
                res = requests.get(url,headers=headers)
                res.encoding = "utf-8"
                soup = BeautifulSoup(res.content,"html.parser")

                newsList = soup.find_all("div",attrs={"class":"box-result clearfix"})
                for news in newsList:
                    try:
                        title = news.select("h2 a")[0].text

                        # In order to prevent the scraping content from being duplicated, add the title of all scraped news articles to
                        # self.title, and then compare each title in the self.title with each title scraped from a news article,
                        # to check whether the article is repeated
                        if title in self.titles:
                            continue
                        else:
                            self.titles.append(title)
                            extract = re.sub(r"(\s{2,}|\n|\r)", "", news.select(".content")[0].text)
                            keyword = None
                            pubtime = news.select("h2 span")[0].text
                            article_url = news.select("h2 a")[0]["href"]
                            # Scrape the content of the article via the sub url of the news article
                            article_content = self.sub_url_scraping(article_url)

                            newsDict = {"title":title,"extract":extract,"keyword":keyword ,"pubtime":pubtime,"article url":article_url,
                                        "article content":article_content}

                            # Add metadata to each article's dictionary to add more complete information to each article
                            newsDict.update(self.metadata)
                            self.contents.append(newsDict)
                            # print(title)

                    except Exception:
                        continue

            except Exception:
                continue


    def sub_url_scraping(self,sub_page_url):
        texts = "" # Texts must be initialized to an empty str, otherwise string concatenation cannot be done below
        try:
            # Scrape the main content of the article according to the sub url of the article atticle
            res = requests.get(sub_page_url)
            # print(res.content)
            # print(res.status_code)
            res.encoding = "utf=8"
            soup = BeautifulSoup(res.content,"html.parser")

            # All the main contents of the article are stored in the text of the "p" tag in the css structure of the web page.
            paragraphs = soup.find_all("p")
            for p in paragraphs:
                try:
                    # Delete multiple spaces, newlines, and whitespace in text paragraphs
                    texts = texts + re.sub(r"(\s{2,}|\n|\r)","",p.text)

                # If an errot occurs in this "p" tag, then ignore it and continue loop
                except Exception:
                    continue

            return texts

        except Exception:
            return None

    # Save scraped articles' contents as csv file
    def save_as_csv(self):
        df = pd.DataFrame(self.contents)
        file_name = self.metadata["main source"] + ".csv"
        df.to_csv(file_name, index=False)


class BBC_China_scraping():

    def __init__(self,term=["中美", "贸易战"], pages=10, main_source = "BBC China News"):
        self.term = term
        self.pages = pages
        self.metadata = {"language":"Chinese","main source":main_source,"source rank":1,"source country":"China"}

        # Store all scraped content into a dictionary list
        self.contents = []
        # In order to prevent the scraping content from being duplicated, add the title of all scraped news articles to
        # self.title, and then compare each title in the self.title with each title scraped from a news article,
        # to check whether the article is repeated
        self.titles = []
        # self.next_page_urls = []

    def MainPage_scraping(self):
        # Matching a keyword variable term with a regular expression is a str variable containing only one keyword
        # or a list variable containing multiple keywords
        term_type = re.search(r"(?<=<class ')\w{1,}(?='>)", str(type(self.term))).group()
        # print(term_type)

        if term_type == "list":
            search_term = " ".join(self.term)
            # Convert the Chinese in the url to utf-8 encoding using the quote function in the urllib.parse library
            search_term_utf8 = quote(search_term)
            # return search_term_utf8

        else:
            # Convert the Chinese in the url to utf-8 encoding using the quote function in the urllib.parse library
            search_term_utf8 = quote(self.term)
            # return search_term_utf8

        # It is also possible to include Chinese directly in the requests.get() method, because the get() method
        # will automatically convert the Chinese in the url to utf-8 encoding.
        # Create a url of all pages directly using the list constructor here
        urls = ["https://www.bbc.com/zhongwen/simp/search/?q=" + search_term_utf8 + "&start="
            + str(page_num * 10 + 1) for page_num in range(0, self.pages)]

        # Using the information of headers of browser to avoid the anti-scraping measure in website
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko'
                                 ') Chrome/65.0.3325.146 Safari/537.36',
                   'Cookie': 'd_c0="AADC7hDPuAuPTl47f3CKGWCcdFaEiAlqUJU=|1494160873"; _zap=948b68c0-7b99-4f88-8d'
                             '86-60bbdfe1d6c0; q_c1=65ef097320d64c8bb8617943d52dd592|1507048007000|1494160866000'
                             '; z_c0="2|1:0|10:1510560275|4:z_c0|92:Mi4xZXFrRkFnQUFBQUFBQU1MdUVNLTRDeVlBQUFCZ0FsV'
                             'k5FNkQyV2dBVVZRSW5pbUlzMHh3VlFCZjd2ak4zM3dRM09n|c0dcda66cfc727d786453bdc3d0a1fc00e3'
                             'cbeb7f34b723755652ed3b07c0016"; __utma=51854390.1508934931.1494160889.1510566336.15'
                             '10566336.5; __utmz=51854390.1510566336.4.3.utmcsr=zhihu.com|utmccn=(referral)|utmcm'
                             'd=referral|utmcct=/question/46417790; __utmv=51854390.100--|2=registration_date=201'
                             '50826=1^3=entry_date=20150826=1; q_c1=65ef097320d64c8bb8617943d52dd592|152083638100'
                             '0|1494160866000; __DAYU_PP=2rVjnUEImjImJIABArJeffffffff858bab885106; aliyungf_tc=AQ'
                             'AAALRrYxtqPgoAPde0AfHsaJjuBFa0; _xsrf=c86800b3-2052-4bf6-a49b-94e513acaa42'}


        # Based on all urls, scraping page by page
        for url in urls:
            try:
                # Based on url, scrape the information from the source.
                # Note, the information of headers of browser needs to be inputted into the requests.get() function,
                # to avoid the anti-scraping measure in website
                res = requests.get(url,headers=headers)
                res.encoding = "utf-8"
                soup = BeautifulSoup(res.content,"html.parser")

                newsList = soup.find_all("div",attrs={"class":"hard-news-unit hard-news-unit--regular faux-block-link"})
                for news in newsList:
                    try:
                        title = news.select(".hard-news-unit__headline a")[0].text

                        # In order to prevent the scraping content from being duplicated, add the title of all scraped news articles to
                        # self.title, and then compare each title in the self.title with each title scraped from a news article,
                        # to check whether the article is repeated
                        if title in self.titles:
                            continue
                        else:
                            self.titles.append(title)
                            extract = re.sub(r"(\s{2,}|\n|\r)", "", news.select(".hard-news-unit__summary")[0].text)
                            keyword = None
                            pubtime = re.sub(r"(\s{1,}|\n|\r)","",news.select(".mini-info-list__item div")[0]["data-datetime"])
                            article_url = news.select(".hard-news-unit__headline a")[0]["href"]

                            # Scrape the content of the article via the sub url of the news article, and delete the useless content in article content
                            article_content = re.sub(r"分享平台FacebookMessengerMessengerTwitter人人网开心网微博QQP"
                                                     r"lurk豆瓣LinkedInWhatsApp复制链接这是外部链接，浏览器将打开另一"
                                                     r"个窗口","",self.sub_url_scraping(article_url))

                            newsDict = {"title":title,"extract":extract,"keyword":keyword ,"pubtime":pubtime,"article url":article_url,
                                        "article content":article_content}

                            # Add metadata to each article's dictionary to add more complete information to each article
                            newsDict.update(self.metadata)
                            self.contents.append(newsDict)
                            # print(title)

                    except Exception:
                        continue

            except Exception:
                continue


    def sub_url_scraping(self,sub_page_url):
        texts = "" # Texts must be initialized to an empty str, otherwise string concatenation cannot be done below
        try:
            # Scrape the main content of the article according to the sub url of the article atticle
            res = requests.get(sub_page_url)
            res.encoding = "utf=8"
            soup = BeautifulSoup(res.content,"html.parser")

            # All the main contents of the article are stored in the text of the "p" tag in the css structure of the web page.
            paragraphs = soup.find_all("p")
            for p in paragraphs:
                try:
                    # Delete multiple spaces, newlines, and whitespace in text paragraphs
                    texts = texts + re.sub(r"(\s{2,}|\n|\r)","",p.text)

                # If an errot occurs in this "p" tag, then ignore it and continue loop
                except Exception:
                    continue

            return texts

        except Exception:
            return None

    # Save scraped articles' contents as csv file
    def save_as_csv(self):
        df = pd.DataFrame(self.contents)
        file_name = self.metadata["main source"] + ".csv"
        df.to_csv(file_name, index=False)


class China_Daily_scraping():

    def __init__(self,term=["中美", "贸易战"], pages=10, main_source = "China Daily News"):
        self.term = term
        self.pages = pages
        self.metadata = {"language":"Chinese","main source":main_source,"source rank":1,"source country":"China"}

        # Store all scraped content into a dictionary list
        self.contents = []
        # In order to prevent the scraping content from being duplicated, add the title of all scraped news articles to
        # self.title, and then compare each title in the self.title with each title scraped from a news article,
        # to check whether the article is repeated
        self.titles = []
        # self.next_page_urls = []

    def MainPage_scraping(self):
        # Matching a keyword variable term with a regular expression is a str variable containing only one keyword
        # or a list variable containing multiple keywords
        term_type = re.search(r"(?<=<class ')\w{1,}(?='>)", str(type(self.term))).group()
        # print(term_type)

        if term_type == "list":
            search_term = " ".join(self.term)
            # Convert the Chinese in the url to utf-8 encoding using the quote function in the urllib.parse library
            search_term_utf8 = quote(search_term)
            # return search_term_utf8

        else:
            # Convert the Chinese in the url to utf-8 encoding using the quote function in the urllib.parse library
            search_term_utf8 = quote(self.term)
            # return search_term_utf8

        # It is also possible to include Chinese directly in the requests.get() method, because the get() method
        # will automatically convert the Chinese in the url to utf-8 encoding.
        # Create a url of all pages directly using the list constructor here
        urls = ["http://newssearch.chinadaily.com.cn/rest/cn/search?keywords=" + search_term_utf8 + "&sort=dp&page="
            + str(page_num) + "&curType=story&type=&channel=&source=" for page_num in range(0, self.pages)]

        # Using the information of headers of browser to avoid the anti-scraping measure in website
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Ge'
                                 'cko) Chrome/75.0.3770.90 Safari/537.36',
                   'Cookie': 'wdcid=21d85ef99e81661b; UM_distinctid=16b86c752e66af-05356e6a204d4-591d3314-144'
                             '000-16b86c752e7512; __auc=4e68e8b316b86c756765ebea874; pt_37a49e8b=uid=yE2Hap8FP'
                             'PlkL8HyViJDxQ&nid=1&vid=O3IL0rVFHZ6-NhKdGPp65w&vn=1&pvn=1&sact=1561334444904&to_'
                             'flag=1&pl=t4NrgYqSK5M357L2nGEQCw*pt*1561334348772; RT="sl=2&ss=1561334335633&tt='
                             '10264&obo=0&sh=1561334348810%3D2%3A0%3A10264%2C1561334341235%3D1%3A0%3A5595&dm=c'
                             'hinadaily.com.cn&si=c3d00e18-f416-4dc4-a0b0-f2007bc3a728&rl=1&bcn=%2F%2F0211c83e'
                             '.akstat.io%2F&r=http%3A%2F%2Fcn.chinadaily.com.cn%2F&ul=1561334781254&hd=1561334781256"'}


        # Based on all urls, scraping page by page
        for url in urls:
            try:
                # Based on url, scrape the information from the source.
                # Note, the information of headers of browser needs to be inputted into the requests.get() function,
                # to avoid the anti-scraping measure in website
                res = requests.get(url,headers=headers)
                res.encoding = "utf-8"
                res = res.json()

                newsList = res["content"]
                # print("The length of newsList:",len(newsList),"\n")
                for news in newsList:
                    try:
                        title = news["title"]

                        # In order to prevent the scraping content from being duplicated, add the title of all scraped news articles to
                        # self.title, and then compare each title in the self.title with each title scraped from a news article,
                        # to check whether the article is repeated
                        if title in self.titles:
                            continue
                        else:
                            self.titles.append(title)
                            try:
                                extract = re.sub(r"(\s{2,}|\n|\r)", "", news["summary"])
                            except Exception:
                                extract = None

                            keyword = " ".join(news["keywords"])  # Use space to connect all keywords in list news["keywords"]
                            pubtime = news["pubDateStr"]
                            article_url = news["url"]
                            # Scrape the content of the article via index in json file directly
                            article_content = news["plainText"]

                            newsDict = {"title":title,"extract":extract,"keyword":keyword ,"pubtime":pubtime,"article url":article_url,
                                        "article content":article_content}

                            # Add metadata to each article's dictionary to add more complete information to each article
                            newsDict.update(self.metadata)
                            self.contents.append(newsDict)
                            # print(title)

                    except Exception:
                        continue

            except Exception:
                continue


    def sub_url_scraping(self,sub_page_url):
        texts = "" # Texts must be initialized to an empty str, otherwise string concatenation cannot be done below
        try:
            # Scrape the main content of the article according to the sub url of the article atticle
            res = requests.get(sub_page_url)
            print(res.status_code)
            res.encoding = "utf=8"
            soup = BeautifulSoup(res.content,"html.parser")

            # All the main contents of the article are stored in the text of the "p" tag in the css structure of the web page.
            paragraphs = soup.find_all("p")
            for p in paragraphs:
                try:
                    # Delete multiple spaces, newlines, and whitespace in text paragraphs
                    texts = texts + re.sub(r"(\s{2,}|\n|\r)","",p.text)

                # If an errot occurs in this "p" tag, then ignore it and continue loop
                except Exception:
                    continue

            return texts

        except Exception:
            return None

    # Save scraped articles' contents as csv file
    def save_as_csv(self):
        df = pd.DataFrame(self.contents)
        file_name = self.metadata["main source"] + ".csv"
        df.to_csv(file_name, index=False)


class ifeng_scraping():

    def __init__(self,term=["中美", "贸易战"], pages=10, main_source = "ifeng News"):
        self.term = term
        self.pages = pages
        self.metadata = {"language":"Chinese","main source":main_source,"source rank":1,"source country":"China"}

        # Store all scraped content into a dictionary list
        self.contents = []
        # In order to prevent the scraping content from being duplicated, add the title of all scraped news articles to
        # self.title, and then compare each title in the self.title with each title scraped from a news article,
        # to check whether the article is repeated
        self.titles = []
        # self.next_page_urls = []

    def MainPage_scraping(self):
        # Matching a keyword variable term with a regular expression is a str variable containing only one keyword
        # or a list variable containing multiple keywords
        term_type = re.search(r"(?<=<class ')\w{1,}(?='>)", str(type(self.term))).group()
        # print(term_type)

        if term_type == "list":
            search_term = " ".join(self.term)
            # Convert the Chinese in the url to utf-8 encoding using the quote function in the urllib.parse library
            search_term_utf8 = quote(search_term)
            # return search_term_utf8

        else:
            # Convert the Chinese in the url to utf-8 encoding using the quote function in the urllib.parse library
            search_term_utf8 = quote(self.term)
            # return search_term_utf8

        # It is also possible to include Chinese directly in the requests.get() method, because the get() method
        # will automatically convert the Chinese in the url to utf-8 encoding.
        # Create a url of all pages directly using the list constructor here
        urls = ["https://search.ifeng.com/sofeng/search.action?q=" + search_term_utf8 + "&c=1&p="
            + str(page_num) for page_num in range(1, self.pages + 1)]

        # Using the information of headers of browser to avoid the anti-scraping measure in website
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko'
                                 ') Chrome/65.0.3325.146 Safari/537.36',
                   'Cookie': 'd_c0="AADC7hDPuAuPTl47f3CKGWCcdFaEiAlqUJU=|1494160873"; _zap=948b68c0-7b99-4f88-8d'
                             '86-60bbdfe1d6c0; q_c1=65ef097320d64c8bb8617943d52dd592|1507048007000|1494160866000'
                             '; z_c0="2|1:0|10:1510560275|4:z_c0|92:Mi4xZXFrRkFnQUFBQUFBQU1MdUVNLTRDeVlBQUFCZ0FsV'
                             'k5FNkQyV2dBVVZRSW5pbUlzMHh3VlFCZjd2ak4zM3dRM09n|c0dcda66cfc727d786453bdc3d0a1fc00e3'
                             'cbeb7f34b723755652ed3b07c0016"; __utma=51854390.1508934931.1494160889.1510566336.15'
                             '10566336.5; __utmz=51854390.1510566336.4.3.utmcsr=zhihu.com|utmccn=(referral)|utmcm'
                             'd=referral|utmcct=/question/46417790; __utmv=51854390.100--|2=registration_date=201'
                             '50826=1^3=entry_date=20150826=1; q_c1=65ef097320d64c8bb8617943d52dd592|152083638100'
                             '0|1494160866000; __DAYU_PP=2rVjnUEImjImJIABArJeffffffff858bab885106; aliyungf_tc=AQ'
                             'AAALRrYxtqPgoAPde0AfHsaJjuBFa0; _xsrf=c86800b3-2052-4bf6-a49b-94e513acaa42'}


        # Based on all urls, scraping page by page
        for url in urls:
            try:
                # Based on url, scrape the information from the source.
                # Note, the information of headers of browser needs to be inputted into the requests.get() function,
                # to avoid the anti-scraping measure in website
                res = requests.get(url,headers=headers)
                res.encoding = "utf-8"
                soup = BeautifulSoup(res.content,"html.parser")

                newsList = soup.find_all("div",attrs={"class":"searchResults"})
                for news in newsList:
                    try:
                        title = re.sub(r"(\s{2,}|\n|\r)","",news.select(".fz16.line24 a")[0].text)

                        # In order to prevent the scraping content from being duplicated, add the title of all scraped news articles to
                        # self.title, and then compare each title in the self.title with each title scraped from a news article,
                        # to check whether the article is repeated
                        if title in self.titles:
                            continue
                        else:
                            self.titles.append(title)
                            extract = None
                            keyword = None
                            pubtime = re.sub(r"(\s{2,}|\n|\r)"," ",news.select("p")[2].text)
                            article_url = news.select(".fz16.line24 a")[0]["href"]

                            # Scrape the content of the article via the sub url of the news article, and delete the useless content in article content
                            article_content = self.sub_url_scraping(article_url)

                            newsDict = {"title":title,"extract":extract,"keyword":keyword ,"pubtime":pubtime,"article url":article_url,
                                        "article content":article_content}

                            # Add metadata to each article's dictionary to add more complete information to each article
                            newsDict.update(self.metadata)
                            self.contents.append(newsDict)
                            # print(title)

                    except Exception:
                        continue

            except Exception:
                continue


    def sub_url_scraping(self,sub_page_url):
        texts = "" # Texts must be initialized to an empty str, otherwise string concatenation cannot be done below
        try:
            # Scrape the main content of the article according to the sub url of the article atticle
            res = requests.get(sub_page_url)
            res.encoding = "utf=8"
            soup = BeautifulSoup(res.content,"html.parser")

            # All the main contents of the article are stored in the text of the "p" tag in the css structure of the web page.
            paragraphs = soup.find_all("p")
            for p in paragraphs:
                try:
                    # Delete multiple spaces, newlines, and whitespace in text paragraphs
                    print(p.text)
                    texts = texts + re.sub(r"(\s{2,}|\n|\r)","",p.text)

                # If an errot occurs in this "p" tag, then ignore it and continue loop
                except Exception:
                    continue

            return texts

        except Exception:
            return None

    # Save scraped articles' contents as csv file
    def save_as_csv(self):
        df = pd.DataFrame(self.contents)
        file_name = self.metadata["main source"] + ".csv"
        df.to_csv(file_name, index=False)


class DT_api_scraping():

    def __init__(self,term=["中美", "贸易战"], main_source = "DT api news"):
        self.term = term
        self.metadata = {"language":"Chinese","main source":main_source,"source rank":1,"source country":"China"}

        # Store all scraped content into a dictionary list
        self.contents = []
        # In order to prevent the scraping content from being duplicated, add the title of all scraped news articles to
        # self.title, and then compare each title in the self.title with each title scraped from a news article,
        # to check whether the article is repeated
        self.titles = []
        # self.next_page_urls = []

    def MainPage_scraping(self):
        # Matching a keyword variable term with a regular expression is a str variable containing only one keyword
        # or a list variable containing multiple keywords
        term_type = re.search(r"(?<=<class ')\w{1,}(?='>)", str(type(self.term))).group()
        # print(term_type)

        if term_type == "list":
            search_term = " ".join(self.term)
        else:
            search_term = self.term

        # It is also possible to include Chinese directly in the requests.get() method, because the get() method
        # will automatically convert the Chinese in the url to utf-8 encoding.
        # Api key is important to connect the api
        api_key = ""
        url = "http://api.avatardata.cn/ActNews/Query?key=" + api_key + "&keyword=" + search_term


        # Based on api url, scraping nees data

        try:
            # Based on url, scrape the information from the source.
            res = requests.get(url)
            res.encoding = "utf-8"
            res.json()
            newsList = res["result"]

            if self.metadata["main source"] != "DT api news":
                for news in newsList:
                    try:
                        if re.search(self.metadata["main source"], news["src"]) is not None:
                            title = news["title"]

                            if title in self.titles:
                                continue
                            else:
                                self.titles.append(title)
                                extract = None
                                keyword = None
                                pubtime = news["pdate_src"]
                                article_url = news["url"]
                                article_content = re.sub(r"(\s{2,}|\n|\r)", "", news["content"])

                                newsDict = {"title": title, "extract": extract, "keyword": keyword, "pubtime": pubtime,
                                            "article url": article_url, "article content": article_content}

                                # Add metadata to each article's dictionary to add more complete information to each article
                                newsDict.update(self.metadata)
                                self.contents.append(newsDict)
                                # print(title)

                        else:
                            continue

                    except Exception:
                        continue

            else:
                for news in newsList:
                    try:
                        title = news["title"]

                        # In order to prevent the scraping content from being duplicated, add the title of all scraped news articles to
                        # self.title, and then compare each title in the self.title with each title scraped from a news article,
                        # to check whether the article is repeated
                        if title in self.titles:
                            continue
                        else:
                            self.titles.append(title)
                            extract = None
                            keyword = None
                            pubtime = news["pdate_src"]
                            article_url = news["url"]
                            article_content = re.sub(r"(\s{2,}|\n|\r)", "", news["content"])

                            newsDict = {"title": title, "extract": extract, "keyword": keyword, "pubtime": pubtime,
                                        "article url": article_url,"article content": article_content}

                            # Add metadata to each article's dictionary to add more complete information to each article
                            newsDict.update(self.metadata)
                            self.contents.append(newsDict)
                            # print(title)

                    except Exception:
                        continue

        except Exception:
            print("API cannot be applied.")
            pass


    def sub_url_scraping(self,sub_page_url):
        texts = "" # Texts must be initialized to an empty str, otherwise string concatenation cannot be done below
        try:
            # Scrape the main content of the article according to the sub url of the article atticle
            res = requests.get(sub_page_url)
            res.encoding = "utf=8"
            soup = BeautifulSoup(res.content,"html.parser")

            # All the main contents of the article are stored in the text of the "p" tag in the css structure of the web page.
            paragraphs = soup.find_all("p")
            for p in paragraphs:
                try:
                    # Delete multiple spaces, newlines, and whitespace in text paragraphs
                    texts = texts + re.sub(r"(\s{2,}|\n|\r)","",p.text)

                # If an errot occurs in this "p" tag, then ignore it and continue loop
                except Exception:
                    continue

            return texts

        except Exception:
            return None

    # Save scraped articles' contents as csv file
    def save_as_csv(self):
        df = pd.DataFrame(self.contents)
        file_name = self.metadata["main source"] + ".csv"
        df.to_csv(file_name, index=False)


# Need to be improved
class People_net_scraping():

    def __init__(self,term=["中美", "贸易战"], pages=10, main_source = "People net News"):
        self.term = term
        self.pages = pages
        self.metadata = {"language":"Chinese","main source":main_source,"source rank":1,"source country":"China"}

        # Store all scraped content into a dictionary list
        self.contents = []
        # In order to prevent the scraping content from being duplicated, add the title of all scraped news articles to
        # self.title, and then compare each title in the self.title with each title scraped from a news article,
        # to check whether the article is repeated
        self.titles = []
        self.next_page_urls = []

    def MainPage_scraping(self):
        # Matching a keyword variable term with a regular expression is a str variable containing only one keyword
        # or a list variable containing multiple keywords
        term_type = re.search(r"(?<=<class ')\w{1,}(?='>)", str(type(self.term))).group()
        # print(term_type)

        if term_type == "list":
            search_term = " ".join(self.term)
            # Convert the Chinese in the url to utf-8 encoding using the quote function in the urllib.parse library
            search_term_utf8 = quote(search_term)
            # return search_term_utf8

        else:
            # Convert the Chinese in the url to utf-8 encoding using the quote function in the urllib.parse library
            search_term_utf8 = quote(self.term)
            # return search_term_utf8

        # It is also possible to include Chinese directly in the requests.get() method, because the get() method
        # will automatically convert the Chinese in the url to utf-8 encoding.
        # Create a url of all pages directly using the list constructor here
        url_first = ["https://www.bbc.com/zhongwen/simp/search/?q=" + search_term_utf8 + "&start="
            + str(page_num) for page_num in range(0, self.pages)]
        self.next_page_urls.append(url_first)

        # Using the information of headers of browser to avoid the anti-scraping measure in website
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko'
                                 ') Chrome/65.0.3325.146 Safari/537.36',
                   'Cookie': 'd_c0="AADC7hDPuAuPTl47f3CKGWCcdFaEiAlqUJU=|1494160873"; _zap=948b68c0-7b99-4f88-8d'
                             '86-60bbdfe1d6c0; q_c1=65ef097320d64c8bb8617943d52dd592|1507048007000|1494160866000'
                             '; z_c0="2|1:0|10:1510560275|4:z_c0|92:Mi4xZXFrRkFnQUFBQUFBQU1MdUVNLTRDeVlBQUFCZ0FsV'
                             'k5FNkQyV2dBVVZRSW5pbUlzMHh3VlFCZjd2ak4zM3dRM09n|c0dcda66cfc727d786453bdc3d0a1fc00e3'
                             'cbeb7f34b723755652ed3b07c0016"; __utma=51854390.1508934931.1494160889.1510566336.15'
                             '10566336.5; __utmz=51854390.1510566336.4.3.utmcsr=zhihu.com|utmccn=(referral)|utmcm'
                             'd=referral|utmcct=/question/46417790; __utmv=51854390.100--|2=registration_date=201'
                             '50826=1^3=entry_date=20150826=1; q_c1=65ef097320d64c8bb8617943d52dd592|152083638100'
                             '0|1494160866000; __DAYU_PP=2rVjnUEImjImJIABArJeffffffff858bab885106; aliyungf_tc=AQ'
                             'AAALRrYxtqPgoAPde0AfHsaJjuBFa0; _xsrf=c86800b3-2052-4bf6-a49b-94e513acaa42'}


        # Based on all urls, scraping page by page
        for url in self.next_page_urls:
            try:
                # Based on url, scrape the information from the source.
                # Note, the information of headers of browser needs to be inputted into the requests.get() function,
                # to avoid the anti-scraping measure in website
                res = requests.get(url,headers=headers)
                res.encoding = "utf-8"
                soup = BeautifulSoup(res.content,"html.parser")

                newsList = soup.find_all("div",attrs={"class":"hard-news-unit hard-news-unit--regular faux-block-link"})
                for news in newsList:
                    try:
                        title = news.select(".hard-news-unit__headline a")[0].text

                        # In order to prevent the scraping content from being duplicated, add the title of all scraped news articles to
                        # self.title, and then compare each title in the self.title with each title scraped from a news article,
                        # to check whether the article is repeated
                        if title in self.titles:
                            continue
                        else:
                            self.titles.append(title)
                            extract = re.sub(r"(\s{2,}|\n|\r)", "", news.select(".hard-news-unit__summary")[0].text)
                            keyword = None
                            pubtime = re.sub(r"(\s{1,}|\n|\r)","",news.select(".mini-info-list__item div")[0]["data-datetime"])
                            article_url = news.select(".hard-news-unit__headline a")[0]["href"]

                            # Scrape the content of the article via the sub url of the news article, and delete the useless content in article content
                            article_content = self.sub_url_scraping(article_url)

                            newsDict = {"title":title,"extract":extract,"keyword":keyword ,"pubtime":pubtime,"article url":article_url,
                                        "article content":article_content}

                            # Add metadata to each article's dictionary to add more complete information to each article
                            newsDict.update(self.metadata)
                            self.contents.append(newsDict)
                            # print(title)

                    except Exception:
                        continue

            except Exception:
                continue


    def sub_url_scraping(self,sub_page_url):
        texts = "" # Texts must be initialized to an empty str, otherwise string concatenation cannot be done below
        try:
            # Scrape the main content of the article according to the sub url of the article atticle
            res = requests.get(sub_page_url)
            res.encoding = "utf=8"
            soup = BeautifulSoup(res.content,"html.parser")

            # All the main contents of the article are stored in the text of the "p" tag in the css structure of the web page.
            paragraphs = soup.find_all("p")
            for p in paragraphs:
                try:
                    # Delete multiple spaces, newlines, and whitespace in text paragraphs
                    texts = texts + re.sub(r"(\s{2,}|\n|\r)","",p.text)

                # If an errot occurs in this "p" tag, then ignore it and continue loop
                except Exception:
                    continue

            return texts

        except Exception:
            return None

    # Save scraped articles' contents as csv file
    def save_as_csv(self):
        df = pd.DataFrame(self.contents)
        file_name = self.metadata["main source"] + ".csv"
        df.to_csv(file_name, index=False)


if __name__ == "__main__":
    # Xinhua_scraper = Xinhua_scraping(term=["中美", "贸易战"], pages=50)
    # Xinhua_scraper.MainPage_scraping()
    # for article in Xinhua_scraper.contents:
    #     print("--------------------------------------------\n",article["article content"])

    Sina_scraper = Sina_scraping(term=["中美", "贸易战"], pages=3)
    Sina_scraper.MainPage_scraping()
    for article in Sina_scraper.contents:
        print("--------------------------------------------\n",article["article content"])

    # BBC_China_scraper = BBC_China_scraping(term=["中美", "贸易战"], pages=2)
    # BBC_China_scraper.MainPage_scraping()
    # for article in BBC_China_scraper.contents:
    #     print("--------------------------------------------\n", article["article content"])

    # China_Daily_scraper = China_Daily_scraping(term=["中美", "贸易战"], pages=2)
    # China_Daily_scraper.MainPage_scraping()
    # for article in China_Daily_scraper.contents:
    #     print("--------------------------------------------\n",article["extract"], article["article content"])


    # ifeng_scraper = ifeng_scraping(term=["中美", "贸易战"], pages=1)
    # ifeng_scraper.MainPage_scraping()
    # for article in ifeng_scraper.contents:
    #     print("--------------------------------------------\n",article["title"], "///" ,article["article content"])

    # DT_api_scraper = DT_api_scraping(term=["中美", "贸易战"])
    # DT_api_scraper.MainPage_scraping()
    # for article in DT_api_scraper.contents:
    #     print("--------------------------------------------\n",article["title"], "///" ,article["article content"])

    # a = "网易"
    # b = "网易新闻"
    # if re.search(a,b) is not None:
    #     print(re.search(a,b).group())
    # else:
    #     print("No match")


    # url = ""
    # content =  Sina_scraping().sub_url_scraping(url)
    # print(content)

