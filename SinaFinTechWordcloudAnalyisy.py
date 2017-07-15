# coding=utf-8
import jieba.posseg as pseg
from wordcloud import WordCloud
from scipy.misc import imread
import requests
import re
import time
from os import path
import matplotlib.pyplot as plt

MAX_PAGE_NUM = 100

def fetch_news(base_url,pattern):
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.8',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.87 Safari/537.36'
    }
    # req = request.Request(request_url,None,headers)
    with open('D:\\workspace\\FileForder\\news\\subject_name.txt','w',encoding='utf-8') as fn :
        with open('D:\\workspace\\FileForder\\news\\subject_url.txt','w',encoding='utf-8') as fu:
            for index in range(1,MAX_PAGE_NUM):
                print('Downloading page #{}'.format(index))
                request_url = base_url + str(index)
                req = requests.get(request_url)
                req.encoding = 'gb2312'
                data = req.text
                datalist = re.findall(pattern,data)
                for li in datalist:
                    fu.write(li[0]+'\n')
                    fn.write(li[1]+'\n')
            time.sleep(5)

def extract_words():
    news_subjects = []
    for li in open('D:\\workspace\\FileForder\\news\\subject_name.txt','r',encoding='utf-8'):
        news_subjects.append(li)

    stop_words = set(
        line.strip() for line in open('D:\\workspace\\FileForder\\news\\stopwords.txt',encoding='gb2312'))
    news_key_list = []
    for subject in news_subjects:
        if subject.isspace():
            continue
        word_list = pseg.cut(subject)
        for word,flag in word_list:
            if not word in stop_words and flag == 'n':
                news_key_list.append(word)
    # d = path.dirname(__file__)
    mask_image = imread(path.join("D:\\workspace\\FileForder\\news\\manky.JPG"))
    content = ' '.join(news_key_list)
    wc = WordCloud(font_path='simhei.ttf', background_color="black",max_words=800)         # 设置字体最大值
    wc.generate(content)
    plt.imshow(wc)
    plt.axis("off")
    wc.to_file('D:\\workspace\\FileForder\\news\\wordcloud.jpg')
    plt.show()

if __name__ == "__main__":
    base_url = "http://roll.news.sina.com.cn/s/channel.php?ch=01#col=89&spec=&type=&ch=01&k=&offset_page=0&offset_num=0&num=60&asc=&page="
    pattern = re.compile(
        r'<span class="c_tit"><a href="(.*?)" target="_blank">(.*?)</a>')
    fetch_news(base_url,pattern)
    extract_words();