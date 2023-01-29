function output = MyFuncion (stringa)
    d = dictionary(character, count);

    for i=1:(size(stringa))
        d(stringa(i)) = 0;
    end

    for i=1:(size(stringa))
        d(stringa(i)) = d(stringa(i)) +1;
    end
    

    index = 0;
    for i=1:(size(stringa))

        index = index + 1 ;

        if(d(stringa(i)) == 1)
            output = index;
        end
        return
    end

end
