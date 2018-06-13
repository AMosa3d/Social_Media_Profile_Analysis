import matplotlib.pyplot as plt
import pyimgur
from collections import Counter
import pdfcrowd


def get_values(Emotional_Res, labels):
    counter = Counter(Emotional_Res)
    values = []
    for i in range(len(labels)):
        values.append(counter[labels[i]])

    return values

def rebuild_lists(labels, values, colors):

    nlabels = []
    nvalues = []
    ncolors = []

    for i in range(len(labels)):
        if values[i] > 0:
            nlabels.append(labels[i])
            nvalues.append(values[i])
            ncolors.append(colors[i])

    return nlabels, nvalues, ncolors

def plot_emotional_function(Emotional_Res):

    labels = ['Neutral', 'Happy', 'Sad', 'Hate', 'Anger']
    values = get_values(Emotional_Res, labels)
    colors = ['green', 'gold', 'gray', 'orange', 'red']

    labels, values, colors = rebuild_lists(labels, values, colors)

    # Plot
    plt.pie(values, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=False, startangle=140,
            wedgeprops={"edgecolor": "k", 'linewidth': 3, 'linestyle': 'solid', 'antialiased': True})

    plt.ylabel('Emotion Ratio')

    plt.axis('equal')
    plt.tight_layout()

    path = 'emotional_plot.png'
    plt.savefig(path)

    return path


def upload_image(image_path):
    client_id = "b4080d81dbe5e9c"

    im = pyimgur.Imgur(client_id)
    plot_image = im.upload_image(image_path, title="Uploading Plot")

    return plot_image.link


def html_creator(avatar_url, handle_str, plot_path,Emotional_Res, Pos_Neg_Res, Keywords, Tweets):

    main_header = """<h1 style="text-align: center;"><strong>Final Statistical Report</strong></h1>"""
    avatar1 = """<p><strong><img style="display: block; margin-left: auto; margin-right: auto;" src=\""""
    avatar2 = """" alt="avatar" width="100" height="100" border="3" /></strong></p>"""
    handle1 = """<h3 style="text-align: center;"><span style="color: #999999;"><strong>"""
    handle2 = """</strong></span></h3>"""
    blank_line = """<p>&nbsp;</p>"""
    plot1 = """<p style="text-align: center;"><strong><img src=\""""
    plot2 = """" alt="plot" width="600" height="400" /></strong></p>"""
    likely_header = """<p style="text-align: center;"><span style="color: #339966;"><strong>""" + handle_str +""" is likely to :&nbsp;</strong></span></p>"""
    keyword1 = """<p style="text-align: center;"><strong>"""
    keyword2 = """</strong></p>"""
    unlikely_header = """<p style="text-align: center;"><span style="color: #ff0000;"><strong>""" + handle_str +""" is unlikely to :&nbsp;</strong></span></p>"""
    states_header = """<p style="text-align: center;"><strong>table of states</strong></p>"""
    table_tag1 = """<table style="margin-left: auto; margin-right: auto;" border="2" cellspacing="3" cellpadding="10">"""
    table_body_tag1 = "<tbody>"
    table_tr1 = "<tr>"
    table_tr2 = "</tr>"
    table_body_tag2 = "</tbody>"
    table_tag2 = "</table> "
    table_headers = """<td>id</td>\n<td>Tweets</td>\n<td>Positive/Negative</td>\n<td>Emotional</td>\n<td>Keywords Extracted</td>"""

    common_pos_keywords, common_neg_keywords = build_dictionary(Keywords,Pos_Neg_Res)

    positive_keywords = ""
    negative_keywords = ""

    for i in range(len(common_pos_keywords)):
        positive_keywords = positive_keywords + keyword1 + common_pos_keywords[i] + keyword2 + "\n"

    for i in range(len(common_neg_keywords)):
        negative_keywords = negative_keywords + keyword1 + common_neg_keywords[i] + keyword2 + "\n"

    table_content = ""

    for i in range(len(Tweets)):
        id_content = "<td>" + str(i+1) + "</td>\n"
        tweet_content = "<td>" + Tweets[i] + "</td>\n"
        pos_neg_content = "<td>" + Pos_Neg_Res[i] + "</td>\n"
        emotional_content = "<td>" + Emotional_Res[i] + "</td>\n"
        keyword_content = "<td>" + Keywords[i] + "</td>\n"
        table_content = table_content + table_tr1 + id_content + tweet_content + pos_neg_content + emotional_content \
                        + keyword_content + table_tr2 + "\n"

    template_file = "<html>" + blank_line + main_header + avatar1 + avatar_url + avatar2 + handle1 + handle_str + handle2 + blank_line + blank_line\
                    + plot1 + plot_path + plot2 + blank_line + likely_header + positive_keywords + blank_line \
                    + unlikely_header + negative_keywords + blank_line + states_header + table_tag1 + table_body_tag1\
                    + table_tr1 + table_headers + table_tr2 + table_content + table_body_tag2 + table_tag2\
                    + blank_line + blank_line + "</html>"

    file_name = "report.html"
    with open(file_name, "w") as text_file:
        print(template_file, file=text_file)

    return file_name


def convert_html_to_image(html_file):
    username = "Mokka47"
    api_key = "9784bd49583665bf62eae49b34748094"
    file_name = ""

    try:
        file_name = 'report.png'
        client = pdfcrowd.HtmlToImageClient(username, api_key)

        client.setOutputFormat('png')

        client.convertFileToFile(html_file, file_name)
        return file_name
    except pdfcrowd.Error as why:
        print(why)
        raise


def main(Tweets, Emotional_Res, Pos_Neg_Res, Keywords, handle, avatar_url):

    plot_path = plot_emotional_function(Emotional_Res)
    plot_path = upload_image(plot_path)
    html_file = html_creator("https://pbs.twimg.com/profile_images/1002238455848595456/YJY8djgO_400x400.jpg",handle,plot_path,Emotional_Res, Pos_Neg_Res, Keywords, Tweets)
    html_image_path = convert_html_to_image(html_file)
    if (html_file == ""):
        return

    report_url = upload_image(html_image_path)

    return report_url

if __name__ == '__main__':
    report = main(['Hello1', 'Hello2', 'Hello3'], ['Neutral', 'Sad', 'Happy', 'Hate', 'Anger', 'Happy'], ['Positive','Positive','Negative'], ['Shady', 'Emam', 'GP'], "@AMosa3d", "avatar_url")
    print(report)