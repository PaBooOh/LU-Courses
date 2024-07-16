
-- explicit join condition in FROM clause
SELECT *
FROM "STUDENTS" "S" JOIN "RESULTS" "R"
ON ("S"."SID" = "R"."SID");

-- implicit join condition in WHERE clause
SELECT *
FROM "STUDENTS" "S", "RESULTS" "R"
WHERE "S"."SID" = "R"."SID";

-- natural join
SELECT *
FROM "STUDENTS" "S" NATURAL JOIN "RESULTS" "R";

-- Theta join
SELECT *
FROM "STUDENTS" "S" JOIN "RESULTS" "R"
ON ("S"."SID" < "R"."SID");

SELECT *
FROM "STUDENTS" "S", "RESULTS" "R"
WHERE "S"."SID" > "R"."SID";
 
-- "forgot" join condition => cross product
SELECT *
FROM "STUDENTS" "S", "RESULTS" "R";

-- multi-attribute join
SELECT *
FROM "STUDENTS" "S" JOIN "RESULTS" "R"
ON "S"."SID" = "R"."SID"
AND "S"."FIRST" LIKE "R"."CAT" || '%';

SELECT *
FROM "STUDENTS" "S" JOIN "RESULTS" "R"
ON "S"."SID" = "R"."SID"
OR SUBSTRING("S"."FIRST",1,1) = "R"."CAT"
ORDER BY "S"."SID", "R"."SID", "R"."CAT";

