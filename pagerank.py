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
from operator import add

from pyspark.sql import SparkSession


def computeContribs(urls, rank):
    """Calculates URL contributions to the rank of other URLs."""
    num_urls = len(urls)
    for url in urls:
        yield (url, rank / num_urls)


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
    #load all urls and the incoming links so we can do 2.2
    linksother  = lines.map(lambda urls: parseNeighborsswapped(urls)).distinct().groupByKey().cache()
    #find all distinct sites in the text file
    union = links.union(linksother)
    # Loads all URLs with other URL(s) link to from input file and initialize ranks of them to one.
    union = union.groupByKey()
    unionfor = union.leftOuterJoin(links)
    union_keys = unionfor.keys().collect()
    # if a siteis a dangling node, then we give it the whole graph as outlinks
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
        #if the site had no incoming links we just set the comtribution to 0 instead of None
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




    total = 0
    # Collects all URL ranks and dump them to console.
    for (link, rank) in ranks.collect():
        print("%s has rank: %s." % (link, rank))
        total+=rank
    #print("#####################################################################")
    #print("total is:",total)
    #print(ranks.count())
    print("total probability is:",total)
    spark.stop()
