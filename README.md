# Introduction and Dataset
This is a team project for the final course project for DATA 144.

We seek to identify the scammers/scamming listings on E-commerce platforms, using machine learning methods. Our project group decided to train our model on Depop, the secondhand ecommerce platform, due to our members’ familiarity with the service and user interface.

In order to detect scam users, our project group collected data from Depop users, both real and fake. We hand-labeled the users according to criteria tabulated in Table 1. The project members identified scam users and products on the website, the details of which were then scraped using Python and the Cloudscraper module, which bypassed any bot detection by Depop. We developed our own scripts for username scraping, and then adopted the product and user variable scraper from github by user Gertje823 (https://github.com/Gertje823/Vinted-Scraper/tree/main). The scraper also included a rate limiter and a sleep time in order to not overwhelm the Depop servers with API requests. The collected data had features about the users such as positive feedback count and the last logged-in time, as well as features describing the various projects on their page such as price, last updated, product descriptions, etc. The full list of both user and product features are tabulated in Table 2. For this project, we annotated 234 users, of which 162 were labeled as established users (Class 0), 12 were labeled as fake users (Class 1), 60 were labeled as new users (Class 2). All products of these users, including selling and sold, sums to a number of 171,910, of which 165,028 were from established users, 4,682 were from fake users, 2,200 were from new users.

# EDA and Methods
After an initial EDA, our project group decided to train the model for classification on a product level, for which the labels were “inherited” from their sellers. We decided to use features that we noticed in our initial runthrough of Depop for scam users that were key identifiers, such as suspiciously low prices, negative reviews, lack of sold products, and so on. By looking at a subset of raw data, we realized that there was an enormous amount of words in product descriptions when combined so we decided to featurize the description with unigrams, which will be discussed further in the individual contribution sections. We tabulated all the features considered by the tree-based models in Table 3. 
 
Table 3. Features Considered by the Tree-based Models

Variable Names
'followers', 'following', 'items_sold', 'reviews_rating',
       'reviews_total', 'size', 'Price', 1000 BOW features

For more details on the tree-based models, please refer to Kai’s individual contribution section.
Following the tree-based models, we implemented a Neural Net with only the product description textual data as the input variable. For the motivations behind this, modeling details, and result interpretations, please refer to Bofei’s individual contribution section.
