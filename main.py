import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()




def main():
    information = """ 
    Elon Reeve Musk (d. 28 Haziran 1971, Pretoria, Güney Afrika), iş adamı, mühendis, endüstriyel tasarımcı, teknoloji girişimcisi ve insanseverdir.[1][2][3][4][5] Elon Musk, eski Amerika Birleşik Devletleri Hükümet Verimliliği Bakanı, SpaceX uzay şirketinin kurucusu, CEO'su ve mühendislik ile tasarım ofislerinin şefi;[6] erken yatırımcı,[7][not 1] Tesla otomotiv şirketinin CEO'su ve ürün mimarı,[10][11] X Corp.'un sahibi, yönetim kurulu başkanı ve CTO'su, The Boring Company ve XAI şirketinin kurucusu,[12] Neuralink, Starlink ile OpenAI'nin kurucu ortağı ve ayrıca ilk eş başkanıdır.[13] Doğduğu yer olan Güney Afrika Cumhuriyeti dışında, Kanada ve ABD vatandaşıdır[14] ve yirmi yaşında göç ettiği ABD'de yaşamaktadır.[15][16]

    Musk, Musk ailesinin üyeleri Kanadalı bir anne ve Güney Afrikalı bir beyaz babanın çocuğu olarak Pretoria, Güney Afrika'da dünyaya geldi ve orada büyüdü.[17] Queen's Üniversitesine gitmek için Kanada'ya taşınmadan önce, kısa süreliğine Pretoria Üniversitesi [en]'ne katıldı.[14] İki yıl sonra Pennsylvania Üniversitesine geçti ve burada Wharton Schooldan[18] ekonomi alanında lisans, fizik alanında ise B.A. ve B.Sc. derecelerini aldı.[19] Doktora derecesine başlamak için 1995 yılında Kaliforniya'ya taşındı ve Stanford Üniversitesinde, uygulamalı fizik ve malzeme bilimleri alanında yüksek lisans yaptı ancak akademik kariyer üzerinden devam etmek yerine iş kariyeri üzerinden devam etmeye karar verdi.[20] 1999'da Compaq tarafından 340 milyon dolara satın alınan bir web yazılım şirketi olan Zip2'yi, kardeşi Kimbal Musk ile birlikte kurdu. Musk, daha sonra çevrimiçi bir banka olan X.com[anlam ayrımı gerekli]'u kurdu. 2000 yılında, bir önceki yıl PayPal'ı kuran ve Ekim 2002'de eBay'e 1,5 milyar dolara satan Confinity ile birleşti.[21][22][23]

    Mayıs 2002'de Musk, günümüzde hâlen CEO'su ve mühendislik ile tasarım ofisleri şefi olduğu, havacılık teknolojisi üreticisi ve uzay taşımacılığı hizmetleri şirketi olan SpaceX'i kurdu. Elektrikli araç üreticisi Tesla Motors, Inc.'e (günümüzdeki Tesla, Inc.) kuruluşundan bir yıl sonra, 2004'te katıldı ve ürün mimarı oldu; 2008'de de şirketin CEO'su oldu. 2006'da, güneş enerjisi hizmetleri şirketi olan SolarCity'nin (günümüzdeki Tesla'nın bir yan kuruluşu) kurulmasına yardımcı oldu. Musk, 2015'te ise dost canlısı olarak gördüğü yapay zekâyı teşvik etmeyi amaçlayan, kâr amacı gütmeyen bir araştırma şirketi olan OpenAI'yi kurdu. Temmuz 2016'da, beyin-bilgisayar arayüzlerini geliştirmeye odaklanmış bir nöroteknoloji şirketi olan Neuralink'i kurdu. Musk, Aralık 2016'da elektrikli araçlar için optimize edilmiş yollara odaklanmış bir "altyapı ve tünel inşaatı" şirketi olan The Boring Company'yi kurdu. Musk, birincil iş arayışlarına ek olarak Hyperloop ismindeki yüksek hızlı bir ulaşım sistemi de tasarladı. Musk, 2018 yılında Kraliyet Topluluğu Üyesi (FRS) seçildi.[24][25] Ayrıca Forbes dergisinin Aralık 2016'da yayımladığı "Dünyanın En Güçlü İnsanları [en]" listesinde 24. sırada;[26] 2019'da, yine Forbes'un yayımladığı "Dünyanın En Yenilikçi İnsanları" listesinde ise ilk sırada yer aldı.[27] Ocak 2021'in ilk haftasında Elon Musk, Jeff Bezos'u geçerek yaşayan en zengin kişi oldu.[28]

    Musk, aynı zamanda alışılmışın dışında duruşlar sergilediği ve çok duyurulan skandallara neden olduğu için de eleştirilere konu oldu. 2018 Tham Luang kurtarma operasyonunda [en] denizaltısı uygun bir seçenek olarak görülmeyip reddedildiğinde Musk, dalgıç takımının liderine "pedo adam" dedi. Dalgıç takımının lideri, Musk'a iftira davası açtı ancak California Hukuk Jürisi, Musk'ın lehine karar verdi. Ayrıca 2018'de Musk, Joe Rogan'ın podcastına kenevir içtiği zamana atıfta bulunarak Tesla'nın özel olarak devralınması için hisse başına 420 dolar fon sağladığını tweetledi. ABD Menkul Kıymetler ve Borsa Komisyonu yorum için kendisine dava açtı, Musk geçici olarak başkanlıktan çekildi ve SEC ile anlaşarak Twitter kullanımındaki sınırlamaları kabul etti. Musk ayrıca yapay zekâ, toplu taşıma ve COVID-19 pandemisi hakkındaki görüşlerinden ötürü de önemli eleştiriler topladı. Musk, Twitter'ı 25 Nisan 2022 tarihinde yaklaşık 44 milyar dolara satın aldı. 9 Temmuz 2022'de ise ihalenin şartlarının delindiğini ifade ederek bu satın alım anlaşmasını feshetti.[29] 2024'te ABD Başkanı Donald Trump tarafından Hükümet Verimliliği bakanı olarak atandı
    """
    
    summary_template = """
    given the information {information} about a person ı want you to create:
    1. A short summary
    2. two interesting facts about them
    """
    
    summary_prompt_template = PromptTemplate(
        input_variables= ["information"], template=summary_template)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    #llm=ChatOllama(temperature=0, model="gemma3:270m")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
