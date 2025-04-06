CREATE TABLE movies (
    MovieName TEXT,
    MovieID INT,
    Genre TEXT,
    UserID INT,
    Rating INT,
    Timestamp BIGINT,
    Gender CHAR(1),
    Age INT,
    Occupation INT,
    Zipcode TEXT,
    age_group TEXT
);

COPY movies(MovieName,MovieID,Genre,UserID,Rating,Timestamp,Gender,Age,Occupation,Zipcode,age_group)
FROM '/docker-entrypoint-initdb.d/movies_cleaned.csv' DELIMITER ',' CSV HEADER;
