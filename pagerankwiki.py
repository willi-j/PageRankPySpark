#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This is an example implementation of PageRank. For more conventional use,
Please refer to PageRank implementation provided by graphx

Example Usage:
bin/spark-submit examples/src/main/python/pagerank.py data/mllib/pagerank_data.txt 10
"""
from __future__ import print_function

import re
import sys
#import matplotlib
from operator import add

from pyspark.sql import SparkSession


def computeContribs(urls, rank):
    """Calculates URL contributions to the rank of other URLs."""
    num_urls = len(urls)
    for url in urls:
        yield (url, rank / num_urls)
def toCSVLine(data):
    return ','.join(str(d) for d in data)


def parseNeighbors(urls):
    """Parses a urls pair string into urls pair."""
    parts = re.split(r'\s+', urls)
    return parts[0], parts[1]

def parseNeighborsswapped(urls):
    """Parses a urls pair string into urls pair."""
    parts = re.split(r'\s+', urls)
    return parts[1], parts[0]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: pagerank <file> <iterations>", file=sys.stderr)
        sys.exit(-1)

    print("WARN: This is a naive implementation of PageRank and is given as an example!\n" +
          "Please refer to PageRank implementation provided by graphx",
          file=sys.stderr)

    # Initialize the spark context.
    spark = SparkSession\
        .builder\
        .appName("PythonPageRank")\
        .getOrCreate()

    # Loads in input file. It should be in format of:
    #     URL         neighbor URL
    #     URL         neighbor URL
    #     URL         neighbor URL
    #     ...
    lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])

    # Loads all URLs from input file and initialize their neighbors.
    links = lines.map(lambda urls: parseNeighbors(urls)).distinct().groupByKey().cache()
    linksother  = lines.map(lambda urls: parseNeighborsswapped(urls)).distinct().groupByKey().cache()
    numoflinks = linksother.map(lambda row: (row[0], len(row[1].data)))
    numoflinks= numoflinks.sortBy(lambda a: -a[1])

    # Loads all URLs with other URL(s) link to from input file and initialize ranks of them to one.
    ranks = links.map(lambda url_neighbors: (url_neighbors[0], 1))
    #n = ranks.count()
    #print("n is :",n,"before iteration")

    union = links.union(linksother)
    union = union.groupByKey()
    unionfor = union.leftOuterJoin(links)
    union_keys = unionfor.keys().collect()

    complete = unionfor.mapValues(lambda a: union_keys if a[1] == None else a[1])
    #print(complete.take(10))

    #print("size on union:", union.count())
    #print("take", union.take(10))
    ranks = complete.map(lambda url_neighbors: (url_neighbors[0], 1.0))
    n = ranks.count()
    ranks = ranks.mapValues(lambda x: x/n)

    # Calculates and updates URL ranks continuously using PageRank algorithm.
    for iteration in range(int(sys.argv[2])):
        # Calculates URL contributions to the rank of other URLs.
        #for (link, rank) in ranks.collect():
        #    print("%s has rank: %s." % (link, rank))
        #print("############################################################################################")
        t_ranks= complete.join(ranks).flatMap(
            lambda url_urls_rank: computeContribs(url_urls_rank[1][0], url_urls_rank[1][1]))
        #print(t_ranks.take(10))
        u_ranks = ranks.leftOuterJoin(t_ranks)
        #print("u_ranks before mapvalues:")
        #for x in u_ranks.collect():
        #    print(x)
        #print("first",u_ranks.take(10))

        u_ranks = u_ranks.mapValues(lambda a: 0 if a[1] == None else a[1])
        #print("u_ranks after mapvalues:")
        #for x in u_ranks.collect():
        #    print(x)
        #print("u_ranks:",u_ranks.take(10))
        # Re-calculates URL ranks based on neighbor contributions.

        ranks = u_ranks.reduceByKey(add).mapValues(lambda rank: rank * 0.85 + 0.15/n)
        #print("link:")
        #for x in links.collect():
        #    print(x[0],end="")
        #    for i in x[1]:
        #        print(" "+i+" ,",end='')
        #    print()
        #print("rank:")
        #for x in ranks.collect():
        #    print(x)




    # Collects all URL ranks and dump them to console.
    newlist = ["Wikipedia","United_States","Windows_XP","Donald_Duck","Ununtrium"]
    count = 0
    ranks= ranks.sortBy(lambda a: -a[1])
    total = 0;
    print("ranks count after iteartion:",ranks.count())
    aggregated = ranks.join(numoflinks)
    aggregated =aggregated.sortBy(lambda a: -a[1][1])
    #printer = aggregated.map(toCSVLine)
    #printer.coalesce(1).saveAsTextFile('sitelinkranks9.csv')

    print("size of links is:", links.count())
    for x in aggregated.take(10):
        print(x)
    for (link, rank) in ranks.collect():
        count +=1
        if link in newlist:
            print("%s has rank: %s." % (link, rank))
        total+=rank
    print("#####################################################################")
    print("total is:",total)
    print(ranks.count())
    print("total probability is:",total)
    print("The Top rank is:",ranks.take(1)[0][0])
    spark.stop()
