from prepare import text_to_int, clean
from machine_learn import model
import pandas as pd


test = pd.read_csv('C:/Users/Radriges/Desktop/Date/response_test.csv')
test_ids = test["client_id"]

test = test.drop(["client_id"], axis=1)
text_to_int(test)
clean(test)

submission_preds = model.predict(test)

itog = pd.DataFrame({"client_id": test_ids.values,
                   "target": submission_preds,
                  })

itog.loc[(itog["target"] < 0.09), 'target'] = int(0)
itog.loc[(itog["target"] >= 0.09), 'target'] = int(1)

itog.to_csv("C:/Users/Radriges/Desktop/Date/itog.csv", index=False, sep=',')
