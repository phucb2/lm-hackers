# utf-8
# encoding: utf-8

# fix encoding
import sys
sys.stdout.reconfigure(encoding='utf-8')

class NumberGenerator:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.zero_to_nine = ['không', 'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín']
        self.zero_to_nine_ = ['', 'mốt', 'hai', 'ba', 'bốn', 'lăm', 'sáu', 'bảy', 'tám', 'chín']
        # dictionary of numbers from 0 to 9
        self.dict_zero_to_nine = dict(zip(range(10), self.zero_to_nine))
        self.dict_zero_to_nine_ = dict(zip(range(10), self.zero_to_nine_))
        
        self.ten_to_million = ['mươi', 'trăm', 'nghìn', 'triệu', 'tỷ']
        self.dict_ten_to_million = dict(zip([10, 100, 1000, 1_000_000, 1_000_000_000], self.ten_to_million))

    def generate(self):
        for i in range(self.start, self.end):
            yield self.convert_to_string(i)
            
    def convert_less_ten(self, number):
        assert number < 10
        return self.dict_zero_to_nine[number]
    
    def convert_between_10_and_100(self, number):
        assert 10 <= number < 100
        if number == 10:
            return 'mười'
        if number % 10 == 0:
            return self.dict_zero_to_nine[number // 10] + ' ' + self.ten_to_million[len(str(number))-2]
        else:
            return self.dict_zero_to_nine[number // 10] + f' {self.ten_to_million[len(str(number))-2]} ' + self.dict_zero_to_nine_[number % 10]
    
    def convert_between_100_and_1000(self, number):
        assert 100 <= number < 1000
        if number == 100:
            return 'một trăm'
        if number % 100 == 0:
            return self.dict_zero_to_nine[number // 100] + ' ' + self.ten_to_million[len(str(number))-2]
        if number % 100 < 10:
            return self.dict_zero_to_nine[number // 100] + ' ' + self.ten_to_million[len(str(number))-2] + ' lẻ ' + self.dict_zero_to_nine[number % 100]
        else:
            return self.dict_zero_to_nine[number // 100] + f' {self.ten_to_million[len(str(number))-2]} ' + self.convert_between_10_and_100(number % 100)
    
    def convert_less_1000(self, number):
        assert number < 1000
        if number < 10:
            return self.convert_less_ten(number)
        elif number < 100:
            return self.convert_between_10_and_100(number)
        else:
            return self.convert_between_100_and_1000(number)
    
    def convert_big_number(self, number):
        # Group number into 3 digits
        # For example: 123456789 -> [[123, 456, 789]]
        groups = []
        while number > 0:
            groups.append(number % 1000)
            number //= 1000
        groups.reverse()
        # Convert each group to string
        # For example: [[123, 456, 789]] -> [['một trăm hai mươi ba', 'bốn trăm năm mươi sáu', 'bảy trăm tám mươi chín']]
        groups = [self.convert_less_1000(g) for g in groups]
        
        # Add 'tỷ', 'triệu', 'nghìn' to each group
        # For example: [['một trăm hai mươi ba', 'bốn trăm năm mươi sáu', 'bảy trăm tám mươi chín']] -> [['một trăm hai mươi ba tỷ', 'bốn trăm năm mươi sáu triệu', 'bảy trăm tám mươi chín nghìn']]
        for i, g in enumerate(groups):
            unit = self.ten_to_million[len(groups) - i] if len(groups) - i > 1 else '' 
            groups[i] = g + ' ' + unit

        
        # Return the result
        # For example: [['một trăm hai mươi ba tỷ', 'bốn trăm năm mươi sáu triệu', 'bảy trăm tám mươi chín nghìn']] -> 'một trăm hai mươi ba tỷ bốn trăm năm mươi sáu triệu bảy trăm tám mươi chín nghìn'
        return ' '.join(groups)  
    
    def convert_to_string(self, number):
        # Convert number to string
        # Split number into list of digits
        digits = list(str(number))
        digits.reverse()
        # Reverse the list
        digits = [int(d) for d in digits]
        # Convert digits to words
        # For example: 123 -> 'ba trăm hai mươi mốt'
        words = []
        for i, d in enumerate(digits):
            if i == 0:
                words.append(self.zero_to_nine[d])
            elif i == 1:
                if d == 0:
                    words.append(self.ten_to_million[i])
                elif d == 1:
                    words.append(self.ten_to_million[i-1])
                else:
                    words.append(self.zero_to_nine[d] + ' ' + self.ten_to_million[i])
            else:
                words.append(self.zero_to_nine[d] + ' ' + self.ten_to_million[i])
        
        return ' '.join(words[::-1])

if __name__ == "__main__":
    # # Generate numbers
    # numbers = NumberGenerator(0, 12)
    # examples = [10, 21, 30, 99]
    # for e in examples:
    #     print(numbers.convert_between_10_and_100(e))
    # examples = [100, 101, 110, 111, 120, 121, 130, 131, 140, 141, 150, 200, 201, 999]
    # for e in examples:
    #     print(numbers.convert_between_100_and_1000(e))
        
    # examples = [1010, 1201, 1_200_300]
    # for e in examples:
    #     print(numbers.convert_big_number(e))
        
    # examples = [1_234_567_909]
    # for e in examples:
    #     print(numbers.convert_big_number(e))

    ng = NumberGenerator(0, 1_000_000_000)
    save_path = 'vn_numbers.txt'
    with open(save_path, 'w') as file:
        for i in range(1_000_000):
            file.write(ng.convert_big_number(i) + '\n')
            if i % 100_000 == 0:
                print(f'Processed {i} numbers')
        
