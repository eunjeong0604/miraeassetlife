

####################################################
# [1] 네이버 뉴스 api 활용해서 대상 리스트 추출
####################################################

def load_news(query='보험 +AI', n=100):
  print('뉴스 가져오기를 시작합니다.')

  # 네이버 API 인증 정보
  client_key = client_id + ':' + client_secret
  client_key_enc = b64encode(client_key.encode('utf-8')).decode('utf-8')

  # API 호출을 위한 URL
  url = 'https://openapi.naver.com/v1/search/news.json'
  headers = {'X-Naver-Client-Id': client_id, 'X-Naver-Client-Secret': client_secret}

  # 검색 요청 파라미터
  params = {
      'query': query,
      'display': 100,  # 검색 결과 중 100개 가져옴(한 번에 최대 100개, 초과 시 오류 반환)
      'sort': 'date',  # 날짜순 정렬(date) 정확도 순 정렬(sim)
      'start': 1
  }

  news_list = []
  for start_idx in range(1, n+1, 10):  # 1부터 100까지 10씩 증가
      params['start'] = start_idx
      response = requests.get(url, headers=headers, params=params)

      # API 응답 확인
      if response.status_code != 200:
          print(f"Failed to get news data at start index {start_idx}: {response.status_code}")
          continue

      news_data = response.json()

      # 'items' 키가 있는지 확인
      if 'items' not in news_data:
          print("No 'items' key in the API response.")
          continue

      # 각 뉴스의 URL과 제목을 가져와서 반환
      for item in news_data['items']:
          news_list.append({
              'title': item['title'],
              'pubDate' : item['pubDate'],
              'url': item['link'],
              'originallink' : item['originallink']
          })
  df = pd.DataFrame(news_list)
  print("--- '{}' 검색 결과, 뉴스 {}건을 가져왔습니다.".format(query, len(news_list)))  # 1000

  ## 네이버뉴스 거르기
  df = df[df['url'].str.contains('https://n.news.naver.com')].reset_index(drop=True)
  print("--- 전체 뉴스 {}건 중 네이버 뉴스는 {}건입니다.".format(len(news_list), len(df)))

  # df.to_excel('네이버뉴스기사크롤링_{}.xlsx'.format(datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Seoul')).strftime('%m%d_%H:%M')), index=False)
  return df



## 언론사 oid 추출 및 matching

def get_oid(df):
  ## 네이버 언론사 뉴스 홈에서 oid 크롤링
  html_company = urllib.request.urlopen('https://news.naver.com/main/officeList.naver').read() #언론사 뉴스 홈 읽기
  soup_company = BeautifulSoup(html_company, 'html.parser')
  title_company = soup_company.find_all(class_='list_press nclicks(\'rig.renws2pname\')')

  cmp_name = []
  cmp_oid = []
  for i in title_company:
    parts = urlparse(i.attrs['href'])
    cmp_name.append(i.get_text().strip())
    cmp_oid.append(parse_qs(parts.query)['officeId'][0])
  oid = pd.DataFrame(list(zip(cmp_name, cmp_oid)), columns=['news_name', 'oid'])
  oid.drop_duplicates(inplace=True) # 중복제거

  ## 크롤링못한 언론사 추가
  add_oid = pd.DataFrame({'news_name':['마이데일리', '스포츠경향', '스포츠서울', '일간스포츠'],
                          'oid':['117', '144', '468', '241']})
  oid = pd.concat([oid, add_oid], ignore_index=True)

  ## 언론사 순위 입력 (2024.05.30 기준)
  # 순위 참고 : https://gobooki.net/%EC%96%B8%EB%A1%A0%EC%82%AC-%EB%84%A4%EC%9D%B4%EB%B2%84-%EC%B1%84%EB%84%90-%EC%B5%9C%EC%8B%A0-%EA%B5%AC%EB%8F%85%EC%9E%90-%EC%88%98-%EC%A0%95%EB%A6%AC/
  news_name_list = [
      "JTBC", "YTN", "MBC", "SBS", "국민일보", "매일경제", "조선일보", "중앙일보", "한국경제", "KBS",
      "한겨레", "경향신문", "동아일보", "머니투데이", "서울경제", "서울신문", "아시아경제", "연합뉴스", "이데일리", "조선비즈",
      "파이낸셜뉴스", "한국일보", "헤럴드경제", "MBN", "뉴스1", "뉴시스", "디지털타임스", "문화일보", "부산일보", "연합뉴스TV",
      "한국경제TV", "노컷뉴스", "SBSBiz", "SBS Biz", "미디어오늘", "kbc광주방송", "TV조선", "강원일보", "데일리안", "매일신문",
      "비즈니스워치", "비즈워치", "세계일보", "시사저널", '마이데일리', "아이뉴스24", "오마이뉴스", "전자신문", "채널A", "프레시안",
      "국제신문", "농민신문", "블로터", "경기일보", "뉴스타파",

      "전주MBC", "조세일보", "강원도민일보", "대전일보", "더팩트",
      "디지털데일리", "매경이코노미", "시사IN", "이코노미스트", "주간조선", "지디넷코리아", "코리아중앙데일리", "코리아헤럴드", "코메디닷컴", "한겨레21",
      "한경비즈니스", "헬스조선",

      '스포츠서울', '일간스포츠', '여성신문', '머니S', '레이디경향', '스포츠경향', '신동아', '주간동아', '동아사이언스',
      '대구MBC', '일다', 'CJB청주방송', '기자협회보', '월간 산', 'JIBS', '더스쿠프', '주간경향', '중앙SUNDAY'
  ]

  news_rank = pd.DataFrame({'news_name': news_name_list, 'news_rank': list(range(1, int(len(news_name_list)) + 1))})
  oid = pd.merge(oid, news_rank, how='left', on='news_name')

  ## 언론사명 매칭
  df['oid'] = df.url.str.split('/').str[5] # 테이블에 언론사 oid 추가
  df = pd.merge(df, oid, how='left', on='oid')
  df.reset_index(drop=True, inplace=True)

  # 언론사명 매칭 안 됐을 경우 예외처리
  df.news_name.fillna('-', inplace=True)
  df.news_rank.fillna(99, inplace=True)

  if len(df[df.news_name=='-'])>0:
    print('--- rank 매칭 안 된 언론사명 : \n', df[df.news_name=='-'][['oid', 'originallink']])
  else:
    print('--- rank 매칭 안 된 언론사명 : 없음')

  return df



####################################################
## [2] 전처리
####################################################

# data cleasing (html 제거)
def clean_html(x):
  x = re.sub("\&\w*\;","", x)  # & ; 형태의 html 값 제거
  x = re.sub("<.*?>","", x) # <> 형태의 html값 제거
  x = re.sub(r'\s+', ' ', x) # 공백 여러개인 경우 1개로 줄여줌
  x = re.sub(r'‘|’|“|”|`|′', "'", x) # ‘’“” -> '로 통일
  x = re.sub(r'&quot;|\.\.\.|…', '', x)
  return x

def news_preprocessing(df):
  print('\n뉴스 전처리를 시작합니다.')

  # 제목 data cleasing (html 제거)
  df['title'] = df['title'].apply(lambda x: clean_html(x))
  print('--- Data Cleansing 완료')

  # 가장 높은 랭킹의 뉴스를 남기고 제목 중복제거
  df = df.sort_values(by='news_rank')
  df.drop_duplicates(subset=['title'], keep='first', inplace=True)
  df.reset_index(drop=True, inplace=True)
  print("--- 가장 높은 랭킹의 뉴스를 남기고 제목 중복제거 후", len(df),"건")

  # 언론사 순위권 이내 기사만 출력
  # ranking=55
  # df = df[df.news_rank <=ranking]
  # print("언론사 순위 "+str(ranking)+"위권 이내 기사 필터링 후 뉴스 개수", len(df))

  # 특정문자 포함 제목 제거
  del_keywords='나스닥|코스피|승진|임명|헤드라인|증시|\(종합\)|표창|기고|공시|주가시황|리포트|.*\[.*\].*|상장|美|中|부고'
  df = df[df.title.str.contains(del_keywords)==False].reset_index(drop=True)
  print("--- 특정문자 포함 제목 제거 후 뉴스 개수", len(df),"건")

  return df



## 뉴스 중복제거
def del_dup_news(df, col):
  # TF-IDF 벡터화 및 코사인 유사도 계산
  vectorizer = TfidfVectorizer()
  tfidf_matrix = vectorizer.fit_transform(df[col])
  cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

  # 유사도에 기반한 군집화
  clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, affinity='cosine', linkage='average')
  clusters = clustering_model.fit_predict(cosine_sim)

  df['cluster'] = clusters
  print('--- TF-IDF기반 군집화 완료')

  # 각 클러스터에서 news_rank가 가장 작은 행만 남기기
  df_dep_dup = df.loc[df.groupby('cluster')['news_rank'].idxmin()].reset_index(drop=True)
  print('--- 같은 cluster 내 언론사 순위가 가장 높은 뉴스만 남기고 중복 제거 후', len(df_dep_dup), '건')

  return df_dep_dup


####################################################
# [3] 주요 뉴스만 뽑기
####################################################

def gpt_imp_news(df):
  global imp_news_yn
  global imp_news_prompt

  print('\n중요한 뉴스를 추출합니다.')

  imp_news_prompt = '''
너는 [기사 제목]을 보고 보험사의 디지털혁신관련 기사인지 분류를 잘하는 분류가야.

아래에 있는 [기사 제목]을 보고 "금융권"의
DX, Digital, 디지털혁신, AI, 생성형AI, 클라우드, 마이데이터, 헬스케어, 요양산업, 실버산업, AICC, 자동화, 어플리케이션
와 관련된 뉴스가 맞으면 1, 아니면 0으로 출력해줘.

반드시 1과 0만 출력해야해
------------------------------------------
[기사 제목] :
'''

  client = OpenAI()
  start_time = time.time()
  imp_news_yn = []
  for i in tqdm(range(len(df))):
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo", # gpt-3.5-turbo
      messages=[
        {"role": "system", "content": "기사 제목을 보고 금융권의 디지털혁신관련 기사인지 분류를 잘하는 분석가"},
        {"role": "user", "content": imp_news_prompt + df['title'][i]}
      ],
      temperature=0, # 최대한 답변을 다르게 하지 않도록 함
      max_tokens=1 # 0, 1만 출력하도록 max token을 낮게 지정
    )
    imp_news_yn.append(completion.choices[0].message.content)
  df['imp_news_yn'] = imp_news_yn
  end_time = time.time()
  print("--- 중요뉴스기사 선별완료 --- 소요된 시간(초):", end_time - start_time)


  ## 중요뉴스기사분류 저장
  no_imp = df[df['imp_news_yn']=='0'].title.sort_values().tolist()
  imp = df[df['imp_news_yn']=='1'].title.sort_values().tolist()
  max_len = max(len(no_imp), len(imp))
  imp_news_yn = pd.DataFrame({
      'no_imp': pd.Series(no_imp + [np.nan] * (max_len - len(no_imp))),
      'imp': pd.Series(imp + [np.nan] * (max_len - len(imp)))
  })
  print('--- 중요기사 {}건, 비중요기사 {}건'.format(len(imp), len(no_imp)))


####################################################
# [4] 해당 뉴스의 전체 본문 크롤링
####################################################

# 추출한 url 주소로 뉴스 전체 본문을 크롤링
def article_full(df):
  print('\n뉴스 본문 크롤링을 시작합니다.')

  start_time = time.time()
  article_full = []
  for news_link in tqdm(df['url']):
    try:
      response = requests.get(news_link)
      soup = BeautifulSoup(response.content, 'html.parser')

      # 네이버 뉴스 기사 본문의 태그는 'div', class 이름은 'article_body'
      article_body = soup.find('div', class_='newsct_article _article_body').get_text()
      article_body = article_body.replace('\t', '').replace('\n', '')
      article_full.append(article_body)
    except Exception as e:
      article_full.append(str(e))
      print("Failed to parse news content:", str(e))

  df['article_full'] = article_full
  end_time = time.time()
  print("--- 뉴스 본문 크롤링 완료 --- 소요된 시간(초):", end_time - start_time)


####################################################
# [4] 해당 뉴스의 전체 본문을 요약하기
####################################################

def gpt_summary(df):
  global summary_prompt
  print('\n전체 본문을 요약합니다.')

  summary_prompt = '''
너는 뉴스기사를 읽고 보험사의 디지털 보고서 제작하는 요약 전문가야.
[제목]은 💡로 시작하고, [주요기업or기관명, 주된 내용] 형식으로 적어줘.

[내용]의 조건은 다음과 같아.
1. -로 시작하고, "명사형"으로 끝내줘.(~함. ~임)
2. 보험사에서 실제 적용할만한 내용 위주로 자세하고 명확하게 요약해줘.
3. [제목]에 있는 단어를 [내용]에 반복하지는 마.

예시:
💡 NH농협생명, '보장분석 자동제공 서비스' 특허 출원
- '보장분석 자동제공 서비스'에 대한 특허 출원을 완료
- 고객에게 보장분석 보고서를 주기적으로 알림톡을 통해 제공하는 내용을 포함, 기존에는 모집인을 통해 제공되던 정보를 고객에게 확대·제공하고자 개발
- 이를 통해 고객은 신계약, 만기, 해지 등에 따른 예상보험금 및 보장현황 변동에 대한 정보를 신속하게 제공 가능
- 농협생명은 특허 출원 내용을 바탕으로 시스템 개발을 진행하며, 내년 중에 해당 서비스를 론칭할 계획


아래에 있는 뉴스기사를 [제목]은 1줄, [내용]은 4~6줄로 요약해줘.

-------------------------------------------------------------------------------------
뉴스기사:
'''

  client = OpenAI()
  start_time = time.time()
  gpt_summary = []
  for i in tqdm(range(len(df))):
      completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
          {"role": "system", "content": "뉴스기사를 읽고 금융 디지털 보고서 제작하는 요약 전문가"},
          {"role": "user", "content": summary_prompt + df['article_full'][i]}
        ],
      temperature=0 # 최대한 답변을 다르게 하지 않도록 함
      )
      gpt_summary.append(completion.choices[0].message.content)
  df['gpt_summary'] = gpt_summary
  end_time = time.time()
  print("--- GPT로 뉴스요약 완료 --- 소요된 시간(초):", end_time - start_time)



####################################################
# [5] 요약 결과 후처리
####################################################

# 최종 요약된 내용을 읽고 기사 중요도 판정
def ranking_imp_news(df):
  global imp_ranking_prompt
  print('\n요약된 뉴스의 중요도를 판별합니다.')

  imp_ranking_prompt='''
너는 [요약된 뉴스기사]를 읽고 뉴스기사의 중요도를 판별하는 판별가야.

"보험사의 디지털 전환 및 신사업 도입"의 관점에서 뉴스기사의 중요도를 숫자 1~5 중 하나를 출력해줘.
뉴스기사가 중요할수록 숫자가 크고, 중요하지 않을수록 숫자가 작아.

예시)
[요약된 뉴스기사]:
보험사의 AI기술을 활용한 고객서비스 업그레이드
- A생명보험사가 decision tree를 이용해 고객분류를 하였다
-> 5

명심해.
너는 1, 2, 3, 4, 5 중에 하나의 숫자를 출력해야해
-------------------------------------------------------------------------------------
[요약된 뉴스기사]:

'''

  client = OpenAI()
  start_time = time.time()
  imp_rank = []
  for i in tqdm(range(len(df))):
      completion = client.chat.completions.create(
        model="gpt-3.5-turbo", # gpt-3.5-turbo
        messages=[
          {"role": "system", "content": "뉴스기사를 읽고 뉴스의 중요도를 판별하는 판별가"},
          {"role": "user", "content": imp_ranking_prompt + df['gpt_summary'][i]}
        ],
      temperature=0, # 최대한 답변을 다르게 하지 않도록 함
      max_tokens=1
      )
      imp_rank.append(completion.choices[0].message.content)
  df['imp_rank'] = imp_rank
  end_time = time.time()
  print("--- 중요뉴스 판별 완료 --- 소요된 시간(초):", end_time - start_time)

# 중요뉴스 top8 이내로 거르기
def del_over8_news(df, max_num=8):
  news_num = len(df)
  if news_num > max_num:
    df = df.sort_values(by='imp_rank').head(8).reset_index(drop=True)
    print(f'--- 최종 기사가 {max_num}건이 넘어, 중요도가 낮은 {news_num}건의 뉴스를 제거하였습니다.')
  else:
    df = df
    print(f'--- 최종 기사가 {max_num}건 이하이므로 뉴스를 제거하지 않았습니다.')


####################################################
# [6] 메일 전송
####################################################

def generate_newsletter(df):
  print('\n요약된 뉴스기사를 메일로 전송합니다.')

  # Jinja2 환경 설정
  env = Environment(loader=FileSystemLoader('.'))
  template = env.get_template('newsletter_template_v2.html')

  # 현재 날짜 가져오기
  current_date = datetime.now().strftime('%Y.%m.%d')

  # 뉴스 맨위 요약 만들기 (news_summary)
  Summary_list = df['gpt_summary']
  html_text = "<p>"
  for summary in Summary_list:
      summary = summary.split('-')[0].strip().replace('💡 ', '👉 ')
      html_text += summary + "<br>"
  html_text += "</p>"
  news_summary = html_text

  # 뉴스 본문 만들기
  df['date'] = pd.to_datetime(df['pubDate'], format='%a, %d %b %Y %H:%M:%S %z').dt.strftime('%Y.%m.%d') # 뉴스레터에 들어갈 날짜 변수 추가
  Date_list = df['date']
  Link_list = df['url']

  news_content = ""
  for summary, link, date in zip(Summary_list, Link_list, Date_list):
      title = summary.split('-')[0].strip()
      formatted_summary = summary.split('-', 1)[1].replace('\n', '<br>')
      news_content += f'''
<p class="news-title">{title}</p>
<p class="news-summary">- {formatted_summary}</p>
<p class="news-date">({date}, <a href="{link}">뉴스 바로가기)</a><br><br></p>


'''

  # 템플릿 렌더링
  output = template.render(current_date=current_date, news_summary=news_summary, news_content=news_content)

  # 결과를 HTML 파일로 저장
  with open('newsletter.html', 'w', encoding='utf-8') as f:
      f.write(output)

  print("--- 뉴스레터가 생성되었습니다.")


# 메일 수신자 리스트 load
def read_recipients_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        recipients = [line.strip() for line in file if line.strip()]
    return recipients


# 뉴스 보내기
def send_newsletter(email_id, email_pw, recipients, subject):
    message = MIMEMultipart()
    message['Subject'] = subject
    message['From'] = f"{email_id}@naver.com"
    message['To'] = ",".join(recipients)

    with open('newsletter.html', 'r', encoding='utf-8') as file:
        html_content = file.read()

    mimetext = MIMEText(html_content, 'html')
    message.attach(mimetext)

    try:
        # 해당 계정의 메일 환경설정에 들어가 IMAP/SMTP 설정 '사용함'으로 저장
        server = smtplib.SMTP('smtp.naver.com', 587)  # 해당 서버의 서버명 및 포트로 변경
        server.ehlo()
        server.starttls()
        server.login(email_id, email_pw)
        server.sendmail(message['From'], recipients, message.as_string())
        server.quit()
        print("\nEmail sent successfully!")
    except Exception as e:
        print(f"\nFailed to send email: {e}")



####################################################
# [7] 최종결과저장
####################################################

def save_result(df):
  print('\n최종 결과를 저장합니다.')
  df['date'] = pd.to_datetime(df['pubDate'], format='%a, %d %b %Y %H:%M:%S %z').dt.strftime('%Y.%m.%d') # 결과 들어갈 날짜 변수 추가
  save_result = df[['title', 'date', 'gpt_summary', 'oid', 'news_name', 'news_rank', 'imp_rank', 'article_full', 'url', 'originallink']]
  prompt_df = pd.DataFrame({
      '중요기사분류 프롬프트' : [imp_news_prompt],
      '뉴스요약 프롬프트' : [summary_prompt],
      '뉴스요약 중요도 선별' : [imp_ranking_prompt]
      }).T

  with pd.ExcelWriter("{}_뉴스요약결과_{}.xlsx".format(query, datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Seoul')).strftime('%m%d_%H:%M'))) as writer:
      save_result.to_excel(writer, sheet_name='뉴스요약결과', index=False)
      imp_news_yn.to_excel(writer, sheet_name='중요뉴스기사분류', index=False)
      prompt_df.to_excel(writer, sheet_name='프롬프트', index=False)
  print('--- 최종결과 저장 시각: ', datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Seoul')).strftime('%m%d_%H:%M'))



####################################################
# 실행
####################################################

if __name__ == "__main__":
  config = configparser.ConfigParser()
  config.read('config.ini')

  ## API Key 설정
  # 네이버 API 인증 정보 / malife 아이디 활용
  client_id = config['DEFAULT']['client_id']
  client_secret = config['DEFAULT']['client_secret']
  # GPT API Key
  os.environ["OPENAI_API_KEY"] = config['DEFAULT']['openai_api_key']

  # Input 값 설정
  query = '보험 +AI' # 검색키워드 (+의 앞은 띄고 뒤는 붙여야 AI를 포함한 검색결과 도출)
  n = 200 # n*10개의 기사추출
  max_num = 8 # 최종 뉴스요약 개수 상한선

  ## [1] 네이버뉴스 크롤링
  df = load_news(query, n)
  df_ = get_oid(df) # 언론사 oid 추출 및 matching

  ## [2] 전처리
  df_filtered = news_preprocessing(df_) # 기본 전처리
  df_filtered = del_dup_news(df_filtered, col='title') # 뉴스 제목 중복제거
  gpt_imp_news(df_filtered) # 중요 뉴스만 뽑기
  df_imp = df_filtered[df_filtered.imp_news_yn=='1'].reset_index(drop=True) # 중요뉴스기사만 최종 테이블에 할당

  ## [3] 전체 본문 크롤링
  article_full(df_imp)

  ## [4] 전체 본문 요약
  gpt_summary(df_imp)

  # [5] 요약 결과 후처리
  df_fin = del_dup_news(df_imp, col='gpt_summary') # 뉴스요약내용 군집화
  ranking_imp_news(df_fin) # 최종 요약된 내용을 읽고 기사 중요도 판정
  del_over8_news(df_fin, max_num=max_num) # 중요뉴스 top8 이내로 거르기

  # [6] 최종결과 메일로 전송
  generate_newsletter(df_fin) # 뉴스레터 만들기(html)
  email_id = os.getenv("EMAIL_ID", config['DEFAULT']['email_id']) # 환경 변수 가져오기 (기본값 설정)
  email_pw = os.getenv("EMAIL_PW", config['DEFAULT']['email_pw'])
  recipients = read_recipients_from_file('recipients_list.txt')
  subject = f"{datetime.today().strftime('%Y년 %m월 %d일')}의 뉴스레터"
  send_newsletter(email_id, email_pw, recipients, subject) # 뉴스레터 보내기

  # [7] 최종결과 저장
  save_result(df_fin)







