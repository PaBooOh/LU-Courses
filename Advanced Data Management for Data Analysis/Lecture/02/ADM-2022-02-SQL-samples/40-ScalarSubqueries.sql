
SELECT DISTINCT "ENO"
FROM "RESULTS"
WHERE "POINTS" = 12;

SELECT MAX("POINTS")
FROM "RESULTS";

SELECT DISTINCT "ENO"
FROM "RESULTS"
WHERE "POINTS" = (
	SELECT MAX("POINTS")
	FROM "RESULTS"
);
