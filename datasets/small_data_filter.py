import os
import json
import re


class DataFilter(object):
    """The data filter process for original data"""

    def __init__(
            self, law_idx_dir, accu_dir,
            filtered_law_idx_dir,
            filtered_accu_dir,
            input_train_file,
            input_test_file,
            input_valid_file):

        self.law_dir = law_idx_dir
        self.accu_dir = accu_dir
        self.filtered_law_dir = filtered_law_idx_dir
        self.filtered_accu_dir = filtered_accu_dir
        self.train_file = input_train_file
        self.test_file = input_test_file
        self.valid_file = input_valid_file

    def _law_to_num(self):
        law_to_nums = {}
        nums_to_law = {}
        law_order = 0
        with open(self.law_dir, 'r') as f_law:
            for law_idx in f_law.readlines():
                law_to_nums[law_idx.strip()] = law_order
                nums_to_law[law_order] = law_idx.strip()
                law_order += 1

        self.law_num = law_order
        return (law_to_nums, nums_to_law)

    def _accu_to_num(self):
        accu_to_nums = {}
        nums_to_accu = {}
        accu_order = 0
        with open(self.accu_dir, 'r') as f_accu:
            for accu in f_accu.readlines():
                accu_to_nums[accu.strip()] = accu_order
                nums_to_accu[accu_order] = accu.strip()
                accu_order += 1

        self.accu_num = accu_order
        return (accu_to_nums, nums_to_accu)

    def _calculate_appear_num(self):
        law_to_nums, nums_to_law = self._law_to_num()
        accu_to_nums, nums_to_accu = self._accu_to_num()
        print("law_num:", self.law_num)
        print("accu num:", self.accu_num)
        law_appear_num = [0] * self.law_num
        accu_appear_num = [0] * self.accu_num
        target_files = [self.train_file, self.valid_file]
        str_pass = '二审'
        for target_file in target_files:
            with open(target_file, 'r', encoding='utf-8') as file_:
                for line in file_.readlines():
                    fact_description = json.loads(line)
                    if (str_pass in fact_description["fact"] != -1 or
                            len(fact_description["meta"]["accusation"]) > 1 or
                            len(fact_description["meta"]["relevant_articles"]) > 1):
                        pass
                    else:
                        law = str(fact_description["meta"]["relevant_articles"][0])
                        accu = fact_description["meta"]["accusation"][0]
                        law_appear_num[law_to_nums[law]] += 1
                        accu_appear_num[accu_to_nums[accu]] += 1
        return (law_appear_num, accu_appear_num, nums_to_law, nums_to_accu, law_to_nums, accu_to_nums)

    def _generage_filtered_law_and_accu(self):
        filtered_law_list = []
        filtered_accu_list = []
        filtered_law_to_num = {}
        filtered_accu_to_num = {}
        law_idx = 0
        accu_idx = 0
        law_appear_num, accu_appear_num, nums_to_law, nums_to_accu, law_to_nums, accu_to_nums = self._calculate_appear_num()

        with open(self.filtered_law_dir, 'w') as filter_law:
            for i in range(self.law_num):
                if law_appear_num[i] >= 100:
                    filtered_law_list.append(i)
                    filtered_law_to_num[str(nums_to_law[i])] = law_idx
                    law_idx += 1
                    filter_law.write(nums_to_law[i] + '\n')

        with open(self.filtered_accu_dir, 'w') as filter_accu:
            for i in range(self.accu_num):
                if accu_appear_num[i] >= 100:
                    filtered_accu_list.append(i)
                    filtered_accu_to_num[str(nums_to_accu[i])] = accu_idx
                    accu_idx += 1
                    filter_accu.write(nums_to_accu[i] + '\n')
        return filtered_law_list, filtered_accu_list, law_to_nums, accu_to_nums, filtered_law_to_num, filtered_accu_to_num

    def fact_process(self):
        filtered_law_list, filtered_accu_list, law_to_nums, accu_to_nums, filtered_law_to_num, filtered_accu_to_num = self._generage_filtered_law_and_accu()
        target_files = [self.train_file, self.test_file, self.valid_file]
        str_pass = '二审'
        longest_imprison = 0
        regex_list = [
            (r"(经审理查明|公诉机关指控|检察院指控|起诉书指控|指控)([，：,:]?)([\s\S]*)([，。,]?)(足以认定|就上述指控|上述事实)", 2),
            (r"(经审理查明|公诉机关指控|检察院指控|起诉书指控|指控)([，：,:]?)([\s\S]*)([，。,]?)(足以认定|就上述指控|上述事实)", 2),
            (r"(经审理查明|公诉机关指控|检察院指控|起诉书指控|指控)([，：,:]?)([\s\S]*)$", 2),
            (r"^([\s\S]*)([，。,]?)(足以认定|就上述指控|上述事实)", 0)]

        for file_ in target_files:
            total_num = 0
            output_file = os.path.dirname(file_) + "/process_" + file_.split('/')[-1]
            process_file = open(output_file, 'w', encoding='utf-8')
            with open(file_, 'r', encoding='utf-8') as target_file_:
                for line in target_file_.readlines():
                    fact_description = json.loads(line)
                    if (str_pass in fact_description["fact"] != -1 or
                            len(fact_description["meta"]["accusation"]) > 1 or
                            len(fact_description["meta"]["relevant_articles"]) > 1):
                        pass
                    else:
                        law = str(fact_description["meta"]["relevant_articles"][0])
                        accu = fact_description["meta"]["accusation"][0]
                        if law_to_nums[law] in filtered_law_list and accu_to_nums[accu] in filtered_accu_list:
                            total_num += 1
                            if fact_description["meta"]["term_of_imprisonment"]["imprisonment"] > longest_imprison:
                                longest_imprison = fact_description["meta"]["term_of_imprisonment"]["imprisonment"]
                            fact = fact_description["fact"]
                            process_fact = fact.replace("b", "").replace("\t", " ").replace("t", "")

                            for reg, num in regex_list:
                                regex = re.compile(reg)
                                result = re.findall(regex, process_fact)
                                if len(result) > 0:
                                    fact = result[0][num]
                                    break

                            new_sample = {}
                            new_sample["fact"] = fact
                            new_sample["accu"] = filtered_accu_to_num[fact_description["meta"]["accusation"][0]]
                            new_sample["law"] = filtered_law_to_num[
                                str(fact_description["meta"]["relevant_articles"][0])]
                            temp_term = fact_description["meta"]["term_of_imprisonment"]
                            new_sample["time"] = temp_term["imprisonment"]
                            new_sample["term_cate"] = 2
                            if temp_term["death_penalty"] == True or temp_term["life_imprisonment"] == True:
                                if temp_term["death_penalty"] == True:
                                    new_sample["term_cate"] = 0 
                                else:
                                    new_sample["term_cate"] = 1
                                new_sample["term"] = 0  # 0 for death_penalty or life imprisonment
                            elif temp_term["imprisonment"] > 10 * 12: # 1 for 10- year (大于10年)
                                new_sample["term"] = 1
                            elif temp_term["imprisonment"] > 7 * 12:  # 2 for 7-10 year (7, 10]
                                new_sample["term"] = 2
                            elif temp_term["imprisonment"] > 5 * 12:  # 3 for 5-7 year (5, 7]
                                new_sample["term"] = 3
                            elif temp_term["imprisonment"] > 3 * 12:  # 4 for 3-5 year (3, 5]
                                new_sample["term"] = 4
                            elif temp_term["imprisonment"] > 2 * 12:  # 5 for 2-3 year (2, 3]
                                new_sample["term"] = 5
                            elif temp_term["imprisonment"] > 1 * 12:  # 6 for 1-2 year (1, 2]
                                new_sample["term"] = 6
                            elif temp_term["imprisonment"] > 9:       # 7 for 0.9-1 year (0.9, 1]
                                new_sample["term"] = 7
                            elif temp_term["imprisonment"] > 6:       # 8 for 0.6-0.9 year (0.6, 0.9]
                                new_sample["term"] = 8
                            elif temp_term["imprisonment"] > 0:       # 9 for 0-0.6 year (0, 0.6]
                                new_sample["term"] = 9
                            else:
                                new_sample["term"] = 10               # 10 for 0 year
                            sample = json.dumps(new_sample, ensure_ascii=False) + '\n'
                            process_file.write(sample)
            process_file.close()


if __name__ == "__main__":
    data_filter = DataFilter("./law.txt", "./accu.txt", "./cail_small/small_law.txt", "./cail_small/small_accu.txt",
                             "./cail_small/small_train.json", "./cail_small/small_test.json",
                             "./cail_small/small_valid.json")
    data_filter.fact_process()
