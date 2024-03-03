import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import hashlib

def process_text(text):
    # remove newlines and extra spaces
    # multiple newlines to one newline
    pattern = r"\n+"
    text = re.sub(pattern, "\n", text)
    # remove leading and trailing spaces
    text = text.strip() 
    return text


class Processor:
    def __init__(self) -> None:
        pass
    
    def run(self, text):
        pass
    
class NoOpProcessor(Processor):
    def run(self, text):
        return text
    
class BasicProcessor(Processor):
    def run(self, text):
        return process_text(text)



def get_hashcode_from_url(url):
    # return md5 from url text
    return hashlib.md5(url.encode()).hexdigest()

def store_html_to_file(html, filename):
    # store html to file
    with open(filename, 'w') as f:
        f.write(html)
        

def simple_crawler(start_url):
    # Send a GET request to the start URL
    response = requests.get(start_url)
    
    # If the request was successful, response.status_code will be 200
    if response.status_code == 200:
        store_html_to_file(response.text, get_hashcode_from_url(start_url))
        # Parse the content of the response using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all the <a> tags in the document
        links = soup.find_all('a')
        
        all_text = soup.get_text()
        print(process_text(all_text))

        # Extract and print each link's href attribute, if it exists
        for link in links:
            href = link.get('href')
            if href:
                # Join the URL with the start URL if it's a relative URL
                full_url = urljoin(start_url, href)
                print(full_url)

import unittest

class TestSimpleCrawler(unittest.TestCase):
    def test_get_hashcode_from_url(self):
        url = 'http://example.com'
        self.assertEqual(get_hashcode_from_url(url), 'b6b9d3bbd7c7e3e81f9b9b7d0f1f4b4f')
        
    def test_process_text(self):
        text = """
        This is a test.
        """
        
        
        

# Example usage
if __name__ == "__main__":
    start_url = 'http://example.com'  # Replace this with the URL you wish to crawl
    simple_crawler(start_url)