Languages and libraries used:

Python: pandas, sklearn.neighbors

Users data are not provided.

Based on analyses made for a dataset of users and the ratings they have assigned to items in their cultural collections, the proposed recommendation system is based on the use of the nearest neighbor algorithm to find the most similar users, sentiment analysis as a decision variable, and basic statistics made on the collection. The premise of the system is simple - to suggest to the user such titles that could be recommended to him by his closest friend during the conversation, of course, based only on the data we have. 
 
The scheme of operation:
1. matching the user with another user with whom he has the most in common.
2. From the base of the similar user, select the titles that he rated as high as possible.
3. check whether the similar user has commented positively on a given title.
4. if so, recommending that title. 
- *If there are more such titles, then check which one has a higher average rating among all users.
- *In case it will be impossible to check the base of the second most similar user, then the third and so on.
- *If there will be no common point then recommending to the user the titles with the best ratio of high ratings to their number (suggestions from the Top list).

The data used was obtained from a web application written specifically for this purpose. 
All items added by users from each of the individual tables were placed in a single master table and collated with the data for all cultural items, so that the final picture of the variables in the table used in this part of the work was as follows:

id - The unique identifier of each record.
user - The name of the user
pos_id - The unique identifier of each user's entry.
title_id - Unique identifier of each cultural work appearing in each database.
type - Type of cultural text.
title - The title of the item.
artist - Artist/author/director. 
genres - Genres of the works.
country - Country of origin of the item.
release_year - The year the work was released.
rating - The rating given by the user.
comment - Comment given to the item by the user.
date_added - The date the item was added to the table.
