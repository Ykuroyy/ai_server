require "net/http"
require "uri"
require "json"

uri = URI.parse("http://localhost:10000/build_cache")
http = Net::HTTP.new(uri.host, uri.port)
request = Net::HTTP::Post.new(uri.request_uri)

response = http.request(request)

puts "📡 ステータス: #{response.code}"
puts "📦 レスポンス: #{response.body}"
