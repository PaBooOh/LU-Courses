
SELECT "FIRST"
FROM "STUDENTS"
WHERE "LAST" = 'Smith'
ORDER BY "FIRST" DESC
LIMIT 10;

SELECT DISTINCT "FIRST"
FROM "STUDENTS"
WHERE "LAST" = 'Smith'
ORDER BY "FIRST" DESC
LIMIT 10;

SELECT "POINTS"
FROM "RESULTS";

SELECT DISTINCT "POINTS"
FROM "RESULTS";

SELECT DISTINCT "POINTS"
FROM "RESULTS"
LIMIT 3;

SELECT "POINTS"
FROM "RESULTS"
ORDER BY "POINTS" ASC;

SELECT DISTINCT "POINTS"
FROM "RESULTS"
ORDER BY "POINTS";

SELECT DISTINCT "POINTS"
FROM "RESULTS"
ORDER BY "POINTS"
LIMIT 3;
