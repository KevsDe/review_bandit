![img](https://i.imgur.com/LuWAlux.jpg)

# The objective of this project: Analyse an Amazon reviews dataset in order to identify low-quality reviews based in contextual and behavioural features.

Information sources:
* [Dataset](https://nijianmo.github.io/amazon/index.html#subsets)

For this project I used:
* Spacy
* TextBlob 
* NLKT
* selectorlib
* Flask
* sklearn
* In adittion to functions that I created during the whole process.

## Summary
For this project, my objective was to identify low-quality reviews based in contextual features such as review length, subjectivity, noun ratio, similarity, sentimental orientation, etc and behavioural features such as review counts, posting rate, etc.

## Valid endpoints:
* / - Home
* /single - Analyse a single review.
* /multiple - Analyse multiple reviews performing web scraping after receiving a valid Amazon link.

## Bibliography
* [Cornell University](https://news.cornell.edu/stories/2011/07/cornell-computers-spot-opinion-spam-online-reviews)
* [Andrew Kruger](https://andrewkruger.github.io/projects/2017-08-25-amazon-reviews)

