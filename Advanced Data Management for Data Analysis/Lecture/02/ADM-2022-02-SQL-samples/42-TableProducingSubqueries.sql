
SELECT "ENO", AVG("POINTS") AS "AV"
FROM "STUDENTS" JOIN "RESULTS" USING ("SID")
WHERE "LAST" = 'Smith'
GROUP BY "ENO"
ORDER BY "AV" LIMIT 10;

		SELECT "LAST", "ENO", "POINTS"
		FROM "STUDENTS" JOIN "RESULTS" USING ("SID");

	SELECT "ENO", "POINTS"
	FROM (
		SELECT "LAST", "ENO", "POINTS"
		FROM "STUDENTS" JOIN "RESULTS" USING ("SID")
	) AS "t0"
	WHERE "LAST" = 'Smith';

SELECT "ENO", AVG("POINTS") AS "AV"
FROM (
	SELECT "ENO", "POINTS"
	FROM (
		SELECT "LAST", "ENO", "POINTS"
		FROM "STUDENTS" JOIN "RESULTS" USING ("SID")
	) AS "t0"
	WHERE "LAST" = 'Smith'
) AS "t1"
GROUP BY "ENO"
ORDER BY "AV"
LIMIT 10;

SELECT "ENO", AVG("POINTS") AS "AV"
FROM "STUDENTS" JOIN "RESULTS" USING ("SID")
WHERE "LAST" = 'Smith'
GROUP BY "ENO"
ORDER BY "AV" LIMIT 10;

