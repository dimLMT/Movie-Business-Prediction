# Movie Business Prediction
For this project, publicly available datasets have been acquired from IMDB (Internet Movie Database) and TMDB (The Movie Database) to conduct an analysis of the factors contributing to a movie's success and to identify which features positively influence a movie's revenue.
![image](https://github.com/dimLMT/Movie-Business-Prediction/assets/36935946/019ef280-541c-43ef-8117-2a47b0ef8d68)

## Part 1: Download IMDB (Internet Movie Database) movie data and filter out the subset of movies requested by the stakeholder.

<a href="https://developer.imdb.com/non-commercial-datasets/#namebasicstsvgz">IMDb Non-Commercial Datasets</a>

Our primary focus is on the following files: title.basics.tsv.gz, title.ratings.tsv.gz, title.akas.tsv.gz. We will carry out data cleaning on these datasets.

## Part 2: Design a MySQL database database for the data and inserting the data.
We create a MySQL database to store the cleaned data.
<img width="800" alt="image" src="https://github.com/dimLMT/Movie-Business-Prediction/assets/36935946/6981752b-eddb-4e5d-ab91-df736b91a957">

There is no financial information included in the IMDB (Internet Movie Database) data, e.g. budget or revenue. The Movie Database (TMDB), however, is a great source of financial data.

## Part 3: Use an API to extract box office financial data and load it into the database.
The <a href="https://www.themoviedb.org/">Movie Database (TMDB)</a> offers a free API for programmatic access to their data.

The API Data Extraction has requested an extraction of the budget, revenue, and MPAA Ratings (G/PG/PG-13/R), also called "Certification".

As a proof-of-concept, we are testing the extraction of movies that were released between 2000 and 2005. We will save the results as separate .csv.gz files and also create a combined .csv.gz file. Additionally, we will store the results in an SQL database for further querying and analysis.

After making API calls, we will take a quick look at the average revenue per certification category.
<img width="600" alt="image" src="https://github.com/dimLMT/Movie-Business-Prediction/assets/36935946/96ba20c5-6c14-4c37-b026-a737f2889cdc">

## Part 4: Apply hypothesis testing to explore what makes a movie "successful".
The stakeholder has also requested statistical tests to obtain mathematically-supported answers to their questions:
- Does the MPAA rating of a movie affect how much revenue the movie generates?
- Do movies that are over 2 hours have a significantly different revenue than movies that under 1.5 hours in length?
- Does the genre of a movie affect how much revenue a movie generates?

For each question, we will present visualizations that support the findings of the statistical tests, such as T-tests and ANOVA.

- The MPAA rating of a movie does affect how much revenue the movie generates.<img width="600" alt="image" src="https://github.com/dimLMT/Movie-Business-Prediction/assets/36935946/c510df43-828b-4d37-9903-fbec36763e5e">
- Movies that are over 2 hours have a significantly different revenue than movies that under 1.5 hours in length.<img width="600" alt="image" src="https://github.com/dimLMT/Movie-Business-Prediction/assets/36935946/cc060b64-4a56-430a-9ada-f45baf913e61">
- The genre of a movie does affect how much revenue a movie generates.<img width="600" alt="image" src="https://github.com/dimLMT/Movie-Business-Prediction/assets/36935946/4a2bb694-d76c-4a50-a99f-b080849972b0">

## Summary and Reccomendations
According to the movie data from 2000-2005, our findings indicate that 'PG,' 'PG-13,' and 'G' certifications tend to have a higher likelihood of revenue success compared to 'R' certification. Additionally, movies longer than 2 hours tend to earn more revenue than those shorter than 1.5 hours. Furthermore, the top four popular movie genres in terms of commercial success are Adventure, Animation, Sci-Fi, and Fantasy.

However, it's important to note that these trends may have evolved in the recent 2020s. Since we are analyzing data from 2000-2005, it is advisable to gather more recent data from the past five years for a more up-to-date analysis.
