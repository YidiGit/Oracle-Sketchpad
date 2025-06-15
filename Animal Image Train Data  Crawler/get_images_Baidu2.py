import requests
import os
import urllib.parse
import re

BASE_URL = 'https://image.baidu.com/search/acjson'
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36",
    "Access-Control-Allow-Credentials": "true",
    "Access-Control-Allow-Origin": "https://m.baidu.com, https://www.baidu.com, http://m.baidu.com,http://www.baidu.com",
    "Connection": "keep-alive",
    "Content-Encoding": "br",
    "Content-Type": "application/json"
}
PAGE_SIZE = 30

def down_load_pics(search_name):
    mkdir_dir_at_curr_path(search_name)
    flag = True
    request_count = 0
    while flag:
        print(f'第 {request_count + 1} 次下载中')
        # get url
        download_num = request_count * PAGE_SIZE
        url = get_url(search_name, download_num)
        print(url)

        try:
            # Send a request to obtain response data.
            resp = requests.get(url, headers=HEADERS)
            resp.raise_for_status()  

            # Manually process the response content.
            content = resp.text
            # # Remove invalid control characters.
            content = re.sub(r'[\x00-\x1f\x7f]', '', content)

            # # Parse JSON.
            try:
                jsonData = resp.json()
                print(jsonData)
            except requests.exceptions.JSONDecodeError:
                import json
                jsonData = json.loads(content)

        except requests.RequestException as e:
            print(f'请求出错: {e}')
            break
        except ValueError as e:
            print(f'解析 JSON 数据出错: {e}')
            break

        # if 'data' not in jsonData or jsonData['data'] == [] or jsonData['data'] == [{}]:
        #     print('已经全部下载完成')
        #     flag = False
        #     break

        #  If data is available, download it.
        for item in jsonData['data']:
            # if 'thumbURL' in item and 'fromPageTitleEnc' in item and search_name in item['fromPageTitleEnc']:
            if 'thumbURL' in item and 'fromPageTitleEnc' in item :
                sub_url = item['thumbURL']
                if sub_url.startswith('http'):
                    try:
                        response = requests.get(sub_url)
                        response.raise_for_status() 
                        
                        file_size = len(os.listdir(search_name))
                        
                        pic_index = file_size + 1
                        
                        curr_file_name = f'{search_name}_{pic_index}'
                        
                        content_type = response.headers.get('Content-Type')
                        if content_type == 'image/jpeg':
                            file_ext = 'jpg'
                        elif content_type == 'image/png':
                            file_ext = 'png'
                        else:
                            file_ext = 'jpg' 

                        # Save the downloaded image data into a folder.

                        with open(os.path.join(search_name, f'{curr_file_name}.{file_ext}'), 'wb') as f:
                            f.write(response.content)
                        print(f'第 {pic_index} 张图片下载完成')
                    except requests.RequestException as e:
                        print(f'error: {e}')

        request_count += 1

def get_url(search_name, page_size):
    url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&logid=8332766429333445053&ipn=rj&ct=201326592&is=&fp=result&fr=&word=' + search_name + '&queryWord=' + search_name + '&cl=2&lm=&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=&hd=&latest=&copyright=&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&expermode=&nojc=&isAsync=&pn=' + str(
        page_size) + '&rn=30&gsm=3c&1721294093333='
    return url

# Create a folder in the current directory.

def mkdir_dir_at_curr_path(dir_name):
    try:
        os.mkdir(dir_name)
        print('folder', dir_name, 'created successfully')
    except FileExistsError:
        print('folder', dir_name, 'existed')

if __name__ == '__main__':
    down_load_pics('十二生肖 狗 简笔画')