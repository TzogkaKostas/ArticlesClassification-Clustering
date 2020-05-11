from sklearn.model_selection import train_test_split
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def create_wordcloud(name, string):
	wordcloud = WordCloud(
		stopwords=STOPWORDS,
		background_color='white', include_numbers=True).generate(string)
	wordcloud.to_file(name)
	fig = plt.figure(1)
	plt.imshow(wordcloud)
	plt.axis('off')
	plt.show()
	fig.savefig(name, dpi=1200)

df_data = pd.read_csv("data.csv", sep='\t')
df_data['content'] = df_data['content'].str.replace('\.|\,', '', regex=True)

create_wordcloud("wordcloud_business.png",
    df_data[df_data['category'] == 'business']['content'].dropna().to_string())

create_wordcloud("wordcloud_entertain.png",
    df_data[df_data['category'] == 'entertainment']['content'].dropna().to_string())

create_wordcloud("wordcloud_politics.png",
    df_data[df_data['category'] == 'politics']['content'].dropna().to_string())

create_wordcloud("wordcloud_sport.png",
    df_data[df_data['category'] == 'sport']['content'].dropna().to_string())

create_wordcloud("wordcloud_tech.png",
    df_data[df_data['category'] == 'tech']['content'].dropna().to_string())