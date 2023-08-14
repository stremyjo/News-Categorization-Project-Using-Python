# Read Me

DATA SHAPE AND STATS: 422937 news pages and divided up into:

152746 	news of business category
108465 	news of science and technology category
115920 	news of business category
45615 	news of health category

2076 clusters of similar news for entertainment category
1789 clusters of similar news for science and technology category
2019 clusters of similar news for business category
1347 clusters of similar news for health category

References to web pages containing a link to one news included in the collection are also included. They are represented as pairs of urls corresponding to 2-page browsing sessions. The collection includes 15516 2-page browsing sessions covering 946 distinct clusters divided up into:

6091 2-page sessions for business category
9425 2-page sessions for entertainment category

# CONTENT

FILENAME #1: newsCorpora.csv (102.297.000 bytes)
DESCRIPTION: News pages
FORMAT: ID \t TITLE \t URL \t PUBLISHER \t CATEGORY \t STORY \t HOSTNAME \t TIMESTAMP

where:
ID		Numeric ID
TITLE		News title
URL		Url
PUBLISHER	Publisher name
CATEGORY	News category (b = business, t = science and technology, e = entertainment, m = health)
STORY		Alphanumeric ID of the cluster that includes news about the same story
HOSTNAME	Url hostname
TIMESTAMP 	Approximate time the news was published, as the number of milliseconds since the epoch 00:00:00 GMT, January 1, 1970

The data for filename #2 has been omitted, as we will not be working with this data set in this project yet (maybe we can tackle it later in a project update 🙂).

But you can find the data for it in the readme file in the dataset download files.