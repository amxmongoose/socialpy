

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import tweepy
import numpy as np
import json
from datetime import datetime

#Set API keys
import os
consumer_key = os.environ["TWIT_KEY"]
consumer_secret = os.environ["TWIT_SECRET"]
access_token = os.environ["ACCESS_TOKEN"]
access_token_secret = os.environ["ACCESS_SECRET"]

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```

#Observed Trend 1: CBS has the most positive average polarity score for their tweets

#Observed Trend 2: Fox News and CNN's average twwet polarity is negative

#Observed Trend 3: NYT's average tweet polarity is closest to neutral


```python
target_users = ("@BBC","@CBS","@CNN","@FoxNews","@nytimes")
sentiment = []
sentiment_df = []

for user in target_users:
    
    oldest_tweet = None
    counter = 1
    
    try:
        public_tweets = api.user_timeline(user,
                                          count=100,
                                          result_type = "recent",
                                          max_id=oldest_tweet)

        for tweet in public_tweets:

            text = tweet["text"]
            time = tweet["created_at"]
            
            results = analyzer.polarity_scores(text)
            
            results["user"] = user
            results["date"] = datetime.strptime(time, '%a %b %d %H:%M:%S %z %Y').date()
            results["time"] = time
            results["tweets ago"] = counter
            
            sentiment.append(results)
            
            counter += 1
            
        oldest_tweet = int(tweet['id_str']) - 1

    except tweepy.TweepError:
        print("Error")
    continue
    
sentiment_df = pd.DataFrame(sentiment)
sentiment_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>compound</th>
      <th>date</th>
      <th>neg</th>
      <th>neu</th>
      <th>pos</th>
      <th>time</th>
      <th>tweets ago</th>
      <th>user</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.6249</td>
      <td>2018-03-03</td>
      <td>0.000</td>
      <td>0.745</td>
      <td>0.255</td>
      <td>Sat Mar 03 21:21:40 +0000 2018</td>
      <td>1</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.2960</td>
      <td>2018-03-03</td>
      <td>0.000</td>
      <td>0.833</td>
      <td>0.167</td>
      <td>Sat Mar 03 19:36:06 +0000 2018</td>
      <td>2</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.1695</td>
      <td>2018-03-03</td>
      <td>0.110</td>
      <td>0.720</td>
      <td>0.170</td>
      <td>Sat Mar 03 19:00:05 +0000 2018</td>
      <td>3</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.7717</td>
      <td>2018-03-03</td>
      <td>0.271</td>
      <td>0.729</td>
      <td>0.000</td>
      <td>Sat Mar 03 18:00:16 +0000 2018</td>
      <td>4</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>2018-03-03</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Sat Mar 03 17:58:00 +0000 2018</td>
      <td>5</td>
      <td>@BBC</td>
    </tr>
  </tbody>
</table>
</div>




```python
sentiment_df.to_csv("sentiment_df.csv",index=False)
```


```python
target_users = ("@BBC","@CBS","@CNN","@FoxNews","@nytimes")
BBC_df = sentiment_df.loc[sentiment_df['user'] == "@BBC"]
CBS_df = sentiment_df.loc[sentiment_df['user'] == "@CBS"]
CNN_df = sentiment_df.loc[sentiment_df['user'] == "@CNN"]
FOX_df = sentiment_df.loc[sentiment_df['user'] == "@FoxNews"]
NYT_df = sentiment_df.loc[sentiment_df['user'] == "@nytimes"]
```


```python
def plot_scatter(handle):
    return plt.scatter(x=handle["tweets ago"], 
                       y=handle["compound"],
                       edgecolors = "black",
                       marker = "o",
                       linewidth = 1.0)

plot_scatter(BBC_df)
plot_scatter(CBS_df)
plot_scatter(CNN_df)
plot_scatter(FOX_df)
plot_scatter(NYT_df)

plt.legend(target_users, 
           title = ('Media Source'),
           loc= 'best', 
           bbox_to_anchor = (1,1), 
           fancybox = True,
           fontsize = 8)

plt.title("Sentiment Analysis of Media Tweets (03/03/2018)")
plt.xlabel('Tweets Ago',fontsize=10)
plt.ylabel('Tweet Polarity',fontsize=10)
plt.xlim(100,0)
plt.ylim(-1,1)        
sea.set()

plt.savefig("Tweet_Sentiment.png")

plt.show
```




    <function matplotlib.pyplot.show>




![png](output_5_1.png)



```python
def plot_bar(handle):
    return plt.bar(x=handle["user"],
                   height = handle["compound"].mean()
                  )

plot_bar(BBC_df)
plot_bar(CBS_df)
plot_bar(CNN_df)
plot_bar(FOX_df)
plot_bar(NYT_df)

plt.legend(target_users, title = ('Media Source'),loc= 'best' , bbox_to_anchor = (1.25,1), fancybox = True,fontsize = 8)
plt.title("Overall Sentiment of Media Source (03/03/2018)")
plt.xlabel('Media Source',fontsize=10)
plt.ylabel('Tweet Polarity',fontsize=10)
plt.ylim(-.4,.4)        
sea.set()


plt.savefig("Source_Sentiment.png")

plt.show
```




    <function matplotlib.pyplot.show>




![png](output_6_1.png)

