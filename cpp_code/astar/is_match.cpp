   #include<iostream>
   #include<vector>

   using namespace std;
   
    bool IsMatch(string s, string p) {
        int m = s.size(), n = p.size();
        vector<vector<bool>> dp(m+1, vector<bool>(n+1)); 
        dp[0][0] = 1;  // 边界条件设置为 true，即空字符可以匹配

        for(int i=0; i<m+1; ++i){
            for(int j=1; j<n+1; ++j){
                if(p[j-1] != '*'){  
                    if(p[j-1] == '.' ||(i>0 && s[i-1] == p[j-1])) // 当前字符匹配
                        dp[i][j] = dp[i-1][j-1];
                        //cout << i << ' ' << j << ' ' << dp[i][j] << endl;
                }else{
                        dp[i][j] = dp[i][j-2];   // 不与 '*' 之前的字母匹配
                        //cout << i << ' ' << j << ' ' << dp[i][j] << endl;
                        if(p[j-2] == '.' || (i>0 && s[i-1] == p[j-2])){    // 匹配'*'之前的字符，相当于不要s[i-1]
                            dp[i][j] = dp[i-1][j] || dp[i][j];
                            //cout << i << ' ' << j << ' ' << dp[i][j] << endl;
                        }
                }
            } // end for j
        } // end for i

        return dp[m][n];
    }
