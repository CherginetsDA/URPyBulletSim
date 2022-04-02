syms t q1(t) q2(t) q3(t) q4(t) q5(t) q6(t) 
q = sym('q', [6,1]);

links(1) = Link('d',0.1625,'a',0,'alpha',pi/2,'m',3.761,'r',[0,-0.02561,0.00193]);
links(2) = Link('d',0,'a',-0.425,'alpha',0,'m',8.058,'r',[0.2125, 0, 0.11336]);
links(3) = Link('d',0,'a',-0.3922,'alpha',0,'m',2.846,'r',[0.15, 0.0, 0.0265]);
links(4) = Link('d',0.1333,'a',0,'alpha',pi/2,'m',1.37,'r',[0, -0.0018, 0.01634]);
links(5) = Link('d',0.0997,'a',0,'alpha',-pi/2,'m',1.3,'r',	[0, 0.0018,0.01634]);
links(6) = Link('d',0.0996,'a',0,'alpha',0,'m',0.365,'r',	[0, 0, -0.001159]);

robot = SerialLink(links, 'name','ur5e');

%%
J = robot.jacobe(q);
Jn = subs(J,q(1),q1);
Jn = subs(Jn,q(2),q2);
Jn = subs(Jn,q(3),q3);
Jn = subs(Jn,q(4),q4);
Jn = subs(Jn,q(5),q5);
Jn = subs(Jn,q(6),q6);
dJ = diff(Jn,t);


sdJ = string(dJ);
fid = fopen('test.txt','wt');
for  i = 1:6
    for j = 1:6
        fprintf(fid,"result["+num2str(i)+"][" + num2str(j)+"] = ");
        fprintf(fid,sdJ(i,j));
        fprintf(fid,'\n');
    end
end
fclose(fid);
disp('Done.');