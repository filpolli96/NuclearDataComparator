from nuclear_data_comparator import *

if __name__ == "__main__":
    comparator = NuclearDataComparator()
    
    element = input("Введите химический элемент для анализа: ").strip()
    comparator.set_element(element)
    
    results = comparator.plot_comparison(
        root_file_dir="/home/polina/server/ZFSRAID/Ing27-HPGe/Nikita/TestTiYield/",
        talys_base_dir="/home/polina/server/ZFSRAID/Ing27-HPGe-LaBr/ComparisonData/dat/",
        external_data_dir="/home/polina/server/RAID1/Articles/DataFromArticle/",
        output_dir="comparison_results"
    )
