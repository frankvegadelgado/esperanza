#                Approximate Independent Set Solver
#                          Frank Vega
#                      December 3rd, 2025

import argparse
import time
import math

from . import algorithm
from . import parser
from . import applogger
from . import utils

def approximate_solution(inputFile, verbose=False, log=False, count=False, bruteForce=False, approximation=False):
    """Finds an approximate Independent Set.

    Args:
        inputFile: Input file path.
        verbose: Enable verbose output.
        log: Enable file logging.
        count: Measure the size of the Independent Set.
        bruteForce: Enable brute force approach.
        approximation: Enable an approximate approach within a ratio of at most polynomial.
    """
    
    logger = applogger.Logger(applogger.FileLogger() if (log) else applogger.ConsoleLogger(verbose))
    # Read and parse a dimacs file
    logger.info(f"Parsing the Input File started")
    started = time.time()
    
    graph = parser.read(inputFile)
    filename = utils.get_file_name(inputFile)
    logger.info(f"Parsing the Input File done in: {(time.time() - started) * 1000.0} milliseconds")
    
    if approximation:
        logger.info("An approximate Solution with a polynomial-approximation ratio started")
        started = time.time()
        
        approximate_result = algorithm.find_independent_set_approximation(graph)

        logger.info(f"An approximate Solution with a polynomial-approximation ratio done in: {(time.time() - started) * 1000.0} milliseconds")
        
        answer = utils.string_result_format(approximate_result, count)
        output = f"{filename}: (approximation) {answer}"
        utils.println(output, logger, log)

    if bruteForce:
        logger.info("A solution with an exponential-time complexity started")
        started = time.time()
        
        brute_force_result = algorithm.find_independent_set_brute_force(graph)

        logger.info(f"A solution with an exponential-time complexity done in: {(time.time() - started) * 1000.0} milliseconds")
        
        answer = utils.string_result_format(brute_force_result, count)
        output = f"{filename}: (Brute Force) {answer}"
        utils.println(output, logger, log)
        
    logger.info("Our Algorithm with an approximate solution started")
    started = time.time()
    
    novel_result = algorithm.find_independent_set(graph)

    logger.info(f"Our Algorithm with an approximate solution done in: {(time.time() - started) * 1000.0} milliseconds")

    answer = utils.string_result_format(novel_result, count)
    output = f"{filename}: {answer}"
    utils.println(output, logger, log)
    if novel_result and (bruteForce or approximation):
        if bruteForce:    
            output = f"Exact Ratio (Optimal/Esperanza): {len(brute_force_result)/len(novel_result)}"
        elif approximation:
            output = f"Approximate Ratio (Approximation/Esperanza): {len(approximate_result)/len(novel_result)}"
        utils.println(output, logger, log)
          
def main():
    
    # Define the parameters
    helper = argparse.ArgumentParser(prog="hope", description='Compute the Approximate Independent Set for undirected graph encoded in DIMACS format.')
    helper.add_argument('-i', '--inputFile', type=str, help='input file path', required=True)
    helper.add_argument('-a', '--approximation', action='store_true', help='enable comparison with a polynomial-time approximation approach within a factor of at most 2')
    helper.add_argument('-b', '--bruteForce', action='store_true', help='enable comparison with the exponential-time brute-force approach')
    helper.add_argument('-c', '--count', action='store_true', help='calculate the size of the Independent Set')
    helper.add_argument('-v', '--verbose', action='store_true', help='anable verbose output')
    helper.add_argument('-l', '--log', action='store_true', help='enable file logging')
    helper.add_argument('--version', action='version', version='%(prog)s 0.0.1')
    
    # Initialize the parameters
    args = helper.parse_args()
    approximate_solution(args.inputFile, 
               verbose=args.verbose, 
               log=args.log,
               count=args.count,
               bruteForce=args.bruteForce,
               approximation=args.approximation)
  

if __name__ == "__main__":
    main()